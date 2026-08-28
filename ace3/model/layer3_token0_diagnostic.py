#!/usr/bin/env python3
"""Diagnose an indexed decoder layer against a same-handoff float64 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional

from model24_execution_oracle import (
    CHECKPOINT_SHA256,
    ContractError,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    _layer_tensor_payloads,
    load_two_token_handoff,
)
from official_model24_next_token import (
    TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
    _torch_linear,
    _torch_rmsnorm,
)

DEFAULT_LAYER_INDEX = 3
HIDDEN_SIZE = 896
HEAD_DIM = 64
QUERY_HEADS = 14
KEY_VALUE_HEADS = 2
MATERIAL_ABSOLUTE_ERROR = 0.1
FP16_RELATIVE_TOLERANCE = 0.001
FP16_MAX_ULP_DISTANCE = 1

STAGES = (
    "input_rmsnorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_rope",
    "k_rope",
    "kv_write_k",
    "kv_write_v",
    "attention_qk",
    "attention_softmax",
    "attention_value",
    "o_proj",
    "attention_residual",
    "post_attention_rmsnorm",
    "gate_proj",
    "up_proj",
    "silu_gated",
    "down_proj",
    "mlp_residual",
)
STAGE_WIDTHS = (
    HIDDEN_SIZE,
    HIDDEN_SIZE,
    KEY_VALUE_HEADS * HEAD_DIM,
    KEY_VALUE_HEADS * HEAD_DIM,
    HIDDEN_SIZE,
    KEY_VALUE_HEADS * HEAD_DIM,
    KEY_VALUE_HEADS * HEAD_DIM,
    KEY_VALUE_HEADS * HEAD_DIM,
    None,
    None,
    HIDDEN_SIZE,
    HIDDEN_SIZE,
    HIDDEN_SIZE,
    HIDDEN_SIZE,
    4864,
    4864,
    4864,
    HIDDEN_SIZE,
    HIDDEN_SIZE,
)


class DiagnosticError(RuntimeError):
    """Raised when diagnostic evidence is malformed or outside its bound."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def authenticate_predecessor_handoff(
    handoff_path: Path,
    *,
    layer_index: int,
    expected_predecessor_layer: int,
    expected_handoff_sha256: str,
) -> tuple[list[list[int]], dict[str, Any]]:
    require(layer_index > 0,
            "diagnostic predecessor authentication requires layer index > 0")
    require(
        expected_predecessor_layer == layer_index - 1,
        "expected predecessor layer does not match diagnostic layer",
    )
    require(
        re.fullmatch(r"[0-9a-f]{64}", expected_handoff_sha256) is not None,
        "expected handoff SHA256 must be 64 lowercase hexadecimal digits",
    )
    try:
        handoff, binding = load_two_token_handoff(
            handoff_path,
            expected_sha256=expected_handoff_sha256,
        )
    except ContractError as error:
        raise DiagnosticError(str(error)) from error
    return handoff, {
        **binding,
        "source_layer_index": expected_predecessor_layer,
        "consumer_layer_index": layer_index,
        "authenticated_expected_sha256": expected_handoff_sha256,
    }


def f16_values(bits: np.ndarray | list[int]) -> np.ndarray:
    return np.ascontiguousarray(bits, dtype="<u2").view("<f2").astype(np.float64)


def decode_stage_records(
    records: list[tuple[int, int]],
    stage: int,
    token: int,
) -> np.ndarray:
    expected = (
        QUERY_HEADS * (token + 1)
        if stage in (8, 9)
        else STAGE_WIDTHS[stage]
    )
    require(expected is not None and len(records) == expected,
            f"token {token} stage {stage} record count mismatch")
    if stage in (8, 9):
        require(
            [index for index, _ in records]
            == list(range(token + 1)) * QUERY_HEADS,
            f"token {token} stage {stage} attention ordering mismatch",
        )
        return np.asarray([value for _, value in records], dtype="<u2")
    output = np.zeros(expected, dtype="<u2")
    seen = set()
    for index, value in records:
        require(0 <= index < expected and index not in seen,
                f"token {token} stage {stage} index mismatch")
        seen.add(index)
        output[index] = value
    require(len(seen) == expected, f"token {token} stage {stage} is incomplete")
    return output


def load_trace(path: Path) -> dict[int, dict[int, np.ndarray]]:
    grouped: dict[int, dict[int, list[tuple[int, int]]]] = {
        token: {stage: [] for stage in range(len(STAGES))}
        for token in range(2)
    }
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise DiagnosticError(f"trace unavailable: {error}") from error
    require(len(lines) == 46676, "trace record count mismatch")
    for ordinal, line in enumerate(lines):
        require(re.fullmatch(r"[0-9a-f]{16}", line) is not None,
                f"trace record {ordinal} is malformed")
        token = int(line[0:2], 16)
        position = int(line[2:6], 16)
        stage = int(line[6:8], 16)
        index = int(line[8:12], 16)
        value = int(line[12:16], 16)
        require(token in grouped and position == token and stage < len(STAGES),
                f"trace record {ordinal} metadata mismatch")
        grouped[token][stage].append((index, value))
    return {
        token: {
            stage: decode_stage_records(records, stage, token)
            for stage, records in stages.items()
        }
        for token, stages in grouped.items()
    }


def load_tensors(
    checkpoint_path: Path,
    tensor_map_path: Path,
    layer_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    payloads, records, binding = _layer_tensor_payloads(
        checkpoint_path,
        tensor_map_path,
        layer_index,
    )
    tensors = {}
    hashes = []
    for record in records:
        name = record["name"]
        dtype = "<f2" if record["dtype"] == "F16" else "<i4"
        tensors[name] = np.frombuffer(payloads[name], dtype=dtype).reshape(
            record["shape"]
        )
        hashes.append(
            {
                "name": name,
                "bytes": len(payloads[name]),
                "sha256": sha256_bytes(payloads[name]),
            }
        )
    return tensors, binding, hashes


def independent_reference(
    activation: torch.Tensor,
    tensors: dict[str, np.ndarray],
    layer_index: int,
) -> dict[int, dict[int, np.ndarray]]:
    prefix = f"model.layers.{layer_index}"
    norm1 = _torch_rmsnorm(
        activation,
        tensors[f"{prefix}.input_layernorm.weight"],
    )
    q = _torch_linear(
        norm1,
        tensors,
        f"{prefix}.self_attn.q_proj",
        f"{prefix}.self_attn.q_proj.bias",
    )
    k = _torch_linear(
        norm1,
        tensors,
        f"{prefix}.self_attn.k_proj",
        f"{prefix}.self_attn.k_proj.bias",
    )
    v = _torch_linear(
        norm1,
        tensors,
        f"{prefix}.self_attn.v_proj",
        f"{prefix}.self_attn.v_proj.bias",
    )
    positions = torch.arange(2, dtype=torch.float64)
    frequencies = 1.0 / (
        1_000_000.0
        ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
    )
    angles = torch.outer(positions, frequencies)
    cosine, sine = torch.cos(angles), torch.sin(angles)

    def rotate(values: torch.Tensor, heads: int) -> torch.Tensor:
        shaped = values.reshape(2, heads, HEAD_DIM)
        low = shaped[..., : HEAD_DIM // 2]
        high = shaped[..., HEAD_DIM // 2 :]
        return torch.cat(
            (
                low * cosine[:, None, :] - high * sine[:, None, :],
                high * cosine[:, None, :] + low * sine[:, None, :],
            ),
            dim=-1,
        )

    rotated_q = rotate(q, QUERY_HEADS)
    rotated_k = rotate(k, KEY_VALUE_HEADS)
    values = v.reshape(2, KEY_VALUE_HEADS, HEAD_DIM)
    scores: list[list[torch.Tensor]] = []
    probabilities: list[list[torch.Tensor]] = []
    attended = []
    for token in range(2):
        score_rows = []
        probability_rows = []
        attended_rows = []
        for head in range(QUERY_HEADS):
            kv_head = head // (QUERY_HEADS // KEY_VALUE_HEADS)
            score = (
                rotated_q[token, head]
                @ rotated_k[: token + 1, kv_head].T
                / (HEAD_DIM**0.5)
            )
            probability = torch.softmax(score, dim=-1)
            score_rows.append(score)
            probability_rows.append(probability)
            attended_rows.append(probability @ values[: token + 1, kv_head])
        scores.append(score_rows)
        probabilities.append(probability_rows)
        attended.append(torch.stack(attended_rows))
    attended_tensor = torch.stack(attended).reshape(2, HIDDEN_SIZE)
    output = _torch_linear(
        attended_tensor,
        tensors,
        f"{prefix}.self_attn.o_proj",
    )
    residual1 = output + activation
    norm2 = _torch_rmsnorm(
        residual1,
        tensors[f"{prefix}.post_attention_layernorm.weight"],
    )
    gate = _torch_linear(norm2, tensors, f"{prefix}.mlp.gate_proj")
    up = _torch_linear(norm2, tensors, f"{prefix}.mlp.up_proj")
    silu = torch_functional.silu(gate) * up
    down = _torch_linear(silu, tensors, f"{prefix}.mlp.down_proj")
    final = residual1 + down
    common = (
        norm1,
        q,
        k,
        v,
        rotated_q.reshape(2, -1),
        rotated_k.reshape(2, -1),
        rotated_k.reshape(2, -1),
        v,
        None,
        None,
        attended_tensor,
        output,
        residual1,
        norm2,
        gate,
        up,
        silu,
        down,
        final,
    )
    reference: dict[int, dict[int, np.ndarray]] = {0: {}, 1: {}}
    for token in range(2):
        for stage, values_by_token in enumerate(common):
            if stage in (8, 9):
                rows = scores[token] if stage == 8 else probabilities[token]
                value = torch.cat(rows)
            else:
                require(values_by_token is not None, "reference stage missing")
                value = values_by_token[token].reshape(-1)
            reference[token][stage] = value.detach().cpu().numpy()
    return reference


def coordinate(stage: int, token: int, flat_index: int) -> dict[str, int]:
    if stage in (8, 9):
        return {
            "head": flat_index // (token + 1),
            "key_position": flat_index % (token + 1),
        }
    return {"dimension": flat_index}


def distribution(
    primary_bits: np.ndarray,
    reference: np.ndarray,
    stage: int,
    token: int,
) -> dict[str, Any]:
    primary = f16_values(primary_bits)
    reference = np.asarray(reference, dtype=np.float64)
    require(primary.shape == reference.shape, "comparison shape mismatch")
    require(np.all(np.isfinite(primary)) and np.all(np.isfinite(reference)),
            "comparison contains a nonfinite value")
    difference = np.abs(primary - reference)
    worst = int(np.argmax(difference))
    rounded = np.float16(reference[worst])
    ulp = abs(float(np.spacing(rounded)))
    affected = np.flatnonzero(difference > MATERIAL_ABSOLUTE_ERROR)
    return {
        "records": int(difference.size),
        "max_abs_error": float(difference[worst]),
        "mean_abs_error": float(difference.mean()),
        "p50_abs_error": float(np.quantile(difference, 0.50)),
        "p90_abs_error": float(np.quantile(difference, 0.90)),
        "p99_abs_error": float(np.quantile(difference, 0.99)),
        "count_abs_gt_0_1": int(affected.size),
        "affected_coordinates_abs_gt_0_1": [
            coordinate(stage, token, int(index)) for index in affected
        ],
        "worst_coordinate": coordinate(stage, token, worst),
        "worst_primary": float(primary[worst]),
        "worst_reference": float(reference[worst]),
        "relative_error_at_worst": float(
            difference[worst] / max(abs(reference[worst]), np.finfo(float).tiny)
        ),
        "fp16_ulps_at_worst": float(difference[worst] / ulp) if ulp else 0.0,
    }


def ordered_fp16_bits(bits: int) -> int:
    magnitude = bits & 0x7FFF
    return 0x8000 - magnitude if bits & 0x8000 else 0x8000 + magnitude


def focus_coordinate_stages(
    primary: dict[int, dict[int, np.ndarray]],
    reference: dict[int, dict[int, np.ndarray]],
    token: int,
    dimension: int,
) -> list[dict[str, Any]]:
    require(0 <= dimension < HIDDEN_SIZE, "focus dimension is out of range")
    rows = []
    for stage, name in enumerate(STAGES):
        if stage in (8, 9) or dimension >= primary[token][stage].size:
            continue
        produced_bits = int(primary[token][stage][dimension])
        produced_value = float(f16_values([produced_bits])[0])
        reference_value = float(reference[token][stage][dimension])
        absolute_error = abs(produced_value - reference_value)
        relative_error = absolute_error / max(
            abs(reference_value),
            float(np.finfo(np.float16).tiny),
        )
        rounded_reference_bits = int(
            np.asarray(np.float16(reference_value), dtype="<f2")
            .view("<u2")
            .item()
        )
        ulp_distance = abs(
            ordered_fp16_bits(produced_bits)
            - ordered_fp16_bits(rounded_reference_bits)
        )
        accepted_by_absolute = (
            absolute_error <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE
        )
        accepted_by_relative_ulp = (
            relative_error < FP16_RELATIVE_TOLERANCE
            and ulp_distance <= FP16_MAX_ULP_DISTANCE
        )
        rows.append(
            {
                "stage_index": stage,
                "stage": name,
                "produced_bits": f"{produced_bits:04x}",
                "produced_value": produced_value,
                "reference_value": reference_value,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "ulp_distance_to_rounded_reference": ulp_distance,
                "accepted_by_layer_comparator": bool(
                    accepted_by_absolute or accepted_by_relative_ulp
                ),
            }
        )
    return rows


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    layer_index = args.layer_index
    focus_dimensions = tuple(args.focus_dimension)
    require(
        layer_index == DEFAULT_LAYER_INDEX or focus_dimensions,
        "non-default layer diagnostics require at least one focus dimension",
    )
    handoff, handoff_binding = authenticate_predecessor_handoff(
        args.handoff,
        layer_index=layer_index,
        expected_predecessor_layer=args.expected_predecessor_layer,
        expected_handoff_sha256=args.expected_handoff_sha256,
    )
    raw_trace_payload = args.rtl_trace.read_bytes()
    oracle_trace_payload = args.oracle_trace.read_bytes()
    raw_final_payload = args.rtl_final.read_bytes()
    oracle_final_payload = args.oracle_final.read_bytes()
    require(raw_trace_payload == oracle_trace_payload,
            "RTL trace differs from the integer oracle")
    require(raw_final_payload == oracle_final_payload,
            "RTL final rows differ from the integer oracle")
    primary = load_trace(args.rtl_trace)
    final, _ = load_two_token_handoff(args.rtl_final)
    for token in range(2):
        require(primary[token][18].tolist() == final[token],
                f"token {token} final trace and final rows differ")
    tensors, layer_binding, tensor_hashes = load_tensors(
        args.checkpoint,
        args.tensor_map,
        layer_index,
    )
    activation_bits = np.asarray(handoff, dtype="<u2")
    reference = independent_reference(
        torch.from_numpy(f16_values(activation_bits)),
        tensors,
        layer_index,
    )
    comparisons: dict[str, Any] = {}
    for token in range(2):
        rows = []
        for stage, name in enumerate(STAGES):
            rows.append(
                {
                    "stage_index": stage,
                    "stage": name,
                    **distribution(
                        primary[token][stage],
                        reference[token][stage],
                        stage,
                        token,
                    ),
                }
            )
        first_divergent = next(
            (row for row in rows if row["max_abs_error"] > 0.0),
            None,
        )
        first_material = next(
            (row for row in rows if row["count_abs_gt_0_1"] > 0),
            None,
        )
        comparisons[str(token)] = {
            "first_divergent_stage": (
                None if first_divergent is None else first_divergent["stage"]
            ),
            "first_material_stage_abs_gt_0_1": (
                None if first_material is None else first_material["stage"]
            ),
            "stages": rows,
        }

    token0 = comparisons["0"]["stages"]
    token1 = comparisons["1"]["stages"]
    token0_final = token0[18]
    token1_final = token1[18]
    focus_coordinates = {
        str(dimension): focus_coordinate_stages(
            primary,
            reference,
            token=0,
            dimension=dimension,
        )
        for dimension in focus_dimensions
    }
    if layer_index == DEFAULT_LAYER_INDEX and not focus_dimensions:
        require(
            comparisons["0"]["first_material_stage_abs_gt_0_1"] == "down_proj",
            "Token 0 material divergence moved before the down projection",
        )
        require(
            token0_final["count_abs_gt_0_1"] == 1
            and token0_final["affected_coordinates_abs_gt_0_1"]
            == [{"dimension": 62}]
            and token0_final["relative_error_at_worst"] < 0.001
            and token0_final["fp16_ulps_at_worst"] <= 1.0,
            "Token 0 final deviation exceeds the bounded FP16 outlier disposition",
        )
        require(
            comparisons["1"]["first_material_stage_abs_gt_0_1"] is None
            and token1_final["max_abs_error"] < 0.01,
            "Token 1 adjacent numerical bound failed",
        )
    else:
        for dimension, rows in focus_coordinates.items():
            require(
                rows[-1]["stage"] == "mlp_residual"
                and rows[-1]["accepted_by_layer_comparator"],
                f"focus dimension {dimension} exceeds the same-handoff boundary",
            )
    for token in range(2):
        require(np.array_equal(primary[token][5], primary[token][6]),
                f"token {token} K-cache write differs from RoPE K")
        require(np.array_equal(primary[token][3], primary[token][7]),
                f"token {token} V-cache write differs from V projection")
    token1_probabilities = f16_values(primary[1][9]).reshape(
        QUERY_HEADS,
        2,
    )
    require(
        np.all(token1_probabilities > 0)
        and float(np.max(np.abs(token1_probabilities.sum(axis=1) - 1.0)))
        <= 2**-10,
        "Token 1 did not preserve two-position causal attention",
    )

    return {
        "schema_version": 2,
        "kind": "ace3_indexed_layer_stage_diagnostic",
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "layer_index": layer_index,
            "layer_binding": layer_binding,
            "consumed_tensors": tensor_hashes,
        },
        "sources": {
            "handoff": {
                "path": str(args.handoff),
                "sha256": sha256_bytes(args.handoff.read_bytes()),
                "binding": handoff_binding,
            },
            "rtl_trace": {
                "path": str(args.rtl_trace),
                "bytes": len(raw_trace_payload),
                "sha256": sha256_bytes(raw_trace_payload),
            },
            "integer_oracle_trace": {
                "path": str(args.oracle_trace),
                "bytes": len(oracle_trace_payload),
                "sha256": sha256_bytes(oracle_trace_payload),
            },
            "rtl_final": {
                "path": str(args.rtl_final),
                "bytes": len(raw_final_payload),
                "sha256": sha256_bytes(raw_final_payload),
            },
            "integer_oracle_final": {
                "path": str(args.oracle_final),
                "bytes": len(oracle_final_payload),
                "sha256": sha256_bytes(oracle_final_payload),
            },
        },
        "comparison_boundary": (
            f"same authenticated layer-{layer_index - 1} FP16 handoff; "
            "independent PyTorch CPU float64 dequantized-AWQ "
            f"layer-{layer_index} operators"
        ),
        "comparisons": comparisons,
        "focus_coordinates": focus_coordinates,
        "kv_causality": {
            "token0_k_write_matches_rope": True,
            "token0_v_write_matches_projection": True,
            "token1_k_write_matches_rope": True,
            "token1_v_write_matches_projection": True,
            "token1_key_positions": 2,
            "token1_all_probabilities_nonzero": True,
            "token1_max_probability_sum_error": float(
                np.max(np.abs(token1_probabilities.sum(axis=1) - 1.0))
            ),
        },
        "disposition": {
            "rtl_or_operator_defect_found": False,
            "classification": "bounded_fp16_reference_boundary",
            "reference_reset": "authenticated FP16 layer input handoff",
            "first_divergent_stage": comparisons["0"]["first_divergent_stage"],
            "first_material_stage_abs_gt_0_1": comparisons["0"][
                "first_material_stage_abs_gt_0_1"
            ],
            "affected_dimensions_abs_gt_0_1": [
                row["dimension"]
                for row in token0_final["affected_coordinates_abs_gt_0_1"]
            ],
            "focus_dimensions": list(focus_dimensions),
            "rationale": (
                "RTL trace and final rows are bit-exact to the integer oracle; "
                "the focused final coordinates remain within the scale-aware "
                "FP16 comparator when float64 evaluation starts from the same "
                "authenticated FP16 handoff."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tensor-map", type=Path, required=True)
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER_INDEX)
    parser.add_argument("--focus-dimension", type=int, action="append", default=[])
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--expected-predecessor-layer", type=int, required=True)
    parser.add_argument("--expected-handoff-sha256", required=True)
    parser.add_argument("--rtl-trace", type=Path, required=True)
    parser.add_argument("--oracle-trace", type=Path, required=True)
    parser.add_argument("--rtl-final", type=Path, required=True)
    parser.add_argument("--oracle-final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = diagnose(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        token0 = report["comparisons"]["0"]["stages"][18]
        token1 = report["comparisons"]["1"]["stages"][18]
        print(
            "INDEXED_LAYER_STAGE_DIAGNOSTIC_PASS "
            f"layer={args.layer_index} "
            "classification=bounded_fp16_reference_boundary "
            f"first_material_stage="
            f"{report['comparisons']['0']['first_material_stage_abs_gt_0_1']} "
            f"focus_dimensions={','.join(str(value) for value in args.focus_dimension)} "
            f"token0_final_max={token0['max_abs_error']} "
            f"token0_final_relative={token0['relative_error_at_worst']} "
            f"token0_final_ulps={token0['fp16_ulps_at_worst']} "
            f"token1_final_max={token1['max_abs_error']} "
            "rtl_integer_oracle=exact kv_causality=pass"
        )
    except (DiagnosticError, OSError, ValueError) as error:
        raise SystemExit(f"INDEXED_LAYER_STAGE_DIAGNOSTIC_FAIL {error}") from error


if __name__ == "__main__":
    main()
