#!/usr/bin/env python3
"""Decompose the preserved layer01 position-2 attention-score mismatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from awq_bit_oracle import AWQ_REVERSE_ORDER
from attention_oracle import HEAD_DIM, QUERY_HEADS, attention_score
from qwen2_rope_oracle import qwen2_coefficient, rotate_pair


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATTEMPT = (
    ROOT / "build/model24_persistent_kv_second_token_attempt002/traversal"
)
DEFAULT_REFERENCE = (
    ROOT / "build/independent_fp16_trajectory_20260906_1133/recovery001"
)
DEFAULT_OUTPUT = (
    ROOT
    / "build/model24_persistent_kv_second_token_layer01_stage08_diagnostic_attempt002"
)
LAYER = 1
POSITION = 2
KEY_VALUE_HEADS = 2
HIDDEN_SIZE = QUERY_HEADS * HEAD_DIM
GROUP_SIZE = 128
ABSOLUTE_TOLERANCE = 0.125
RELATIVE_TOLERANCE = 0.001
MAX_ULP_DISTANCE = 1


class DiagnosticError(RuntimeError):
    """Raised when preserved inputs or the controlled comparison are invalid."""


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


def authenticated_trace(
    comparison_path: Path,
    trace_path: Path,
) -> bytes:
    comparison = load_json(comparison_path)
    expected = comparison.get("binding", {}).get("trace.hex", {})
    actual = file_record(trace_path)
    require(
        actual["bytes"] == expected.get("bytes")
        and actual["sha256"] == expected.get("sha256"),
        f"preserved trace binding mismatch: {trace_path}",
    )
    return trace_path.read_bytes()


def trace_stage(payload: bytes, position: int, stage: int) -> np.ndarray:
    records: list[tuple[int, int]] = []
    for ordinal, row in enumerate(payload.decode("ascii").splitlines()):
        require(len(row) == 16, f"trace row {ordinal} width mismatch")
        require(
            int(row[0:2], 16) == 0 and int(row[2:6], 16) == position,
            f"trace row {ordinal} owner mismatch",
        )
        if int(row[6:8], 16) == stage:
            records.append((int(row[8:12], 16), int(row[12:16], 16)))
    if stage not in (8, 9):
        records.sort()
        require(
            [index for index, _ in records] == list(range(len(records))),
            f"trace stage {stage} index mismatch",
        )
    return np.asarray([value for _, value in records], dtype="<u2")


def authenticated_array(record: dict[str, Any]) -> np.ndarray:
    path = Path(record["path"])
    require(
        sha256_bytes(path.read_bytes()) == record["file_sha256"],
        f"independent array file binding mismatch: {path}",
    )
    value = np.load(path, allow_pickle=False)
    require(
        value.dtype == np.dtype("<u2")
        and sha256_bytes(np.ascontiguousarray(value).tobytes())
        == record["semantic_sha256"],
        f"independent array semantic binding mismatch: {path}",
    )
    return value


def load_hex(path: Path, digits: int) -> np.ndarray:
    rows = path.read_text(encoding="ascii").splitlines()
    require(
        rows and all(len(row) == digits for row in rows),
        f"malformed hexadecimal tensor: {path}",
    )
    dtype = "<u2" if digits == 4 else "<u4"
    return np.asarray([int(row, 16) for row in rows], dtype=dtype)


def binary64_q_projection(
    activation_bits: np.ndarray,
    tensor_dir: Path,
) -> np.ndarray:
    require(
        activation_bits.shape == (HIDDEN_SIZE,),
        "Q projection activation geometry mismatch",
    )
    packed_words = HIDDEN_SIZE // 8
    groups = HIDDEN_SIZE // GROUP_SIZE
    qweight = load_hex(
        tensor_dir / "layer1_self_attn_q_proj_qweight.i32le.bin.hex",
        8,
    ).reshape(HIDDEN_SIZE, packed_words)
    qzeros = load_hex(
        tensor_dir / "layer1_self_attn_q_proj_qzeros.i32le.bin.hex",
        8,
    ).reshape(groups, packed_words)
    scales = load_hex(
        tensor_dir / "layer1_self_attn_q_proj_scales.fp16le.bin.hex",
        4,
    ).view("<f2").astype(np.float64).reshape(groups, HIDDEN_SIZE)
    bias = load_hex(
        tensor_dir / "layer1_self_attn_q_proj_bias.fp16le.bin.hex",
        4,
    ).view("<f2").astype(np.float64)

    quantized = np.empty((HIDDEN_SIZE, HIDDEN_SIZE), dtype=np.float64)
    zeros = np.empty((groups, HIDDEN_SIZE), dtype=np.float64)
    for channel in range(HIDDEN_SIZE):
        packed, lane = divmod(channel, 8)
        shift = 4 * AWQ_REVERSE_ORDER[lane]
        quantized[:, channel] = (qweight[:, packed] >> shift) & 0xF
        zeros[:, channel] = (qzeros[:, packed] >> shift) & 0xF
    weight = (
        quantized - np.repeat(zeros, GROUP_SIZE, axis=0)
    ) * np.repeat(scales, GROUP_SIZE, axis=0)
    activation = torch.from_numpy(
        activation_bits.view("<f2").astype(np.float64)
    )
    output = activation @ torch.from_numpy(weight) + torch.from_numpy(bias)
    return np.asarray(output.numpy(), dtype="<f2").view("<u2")


def reference_stage(
    layer_record: dict[str, Any],
    stage: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    suffix = f"/layer01/position002/stage{stage:02d}.npy"
    matches = [
        record
        for record in layer_record["stages"]
        if str(record["path"]).endswith(suffix)
    ]
    require(len(matches) == 1, f"reference stage {stage} record mismatch")
    return authenticated_array(matches[0]), matches[0]


def score_matrix(q_bits: np.ndarray, k_bits: np.ndarray) -> np.ndarray:
    require(q_bits.shape == (QUERY_HEADS * HEAD_DIM,), "Q geometry mismatch")
    require(
        k_bits.shape == (POSITION + 1, KEY_VALUE_HEADS, HEAD_DIM),
        "K cache geometry mismatch",
    )
    scores = []
    for query_head in range(QUERY_HEADS):
        kv_head = query_head // (QUERY_HEADS // KEY_VALUE_HEADS)
        q_head = q_bits[
            query_head * HEAD_DIM : (query_head + 1) * HEAD_DIM
        ].tolist()
        for key_position in range(POSITION + 1):
            result = attention_score(
                q_head,
                k_bits[key_position, kv_head].tolist(),
                [True] * HEAD_DIM,
                POSITION,
                key_position,
            )
            require(
                not result.invalid
                and not result.cache_miss
                and not result.saturation,
                "controlled score evaluation failed",
            )
            scores.append(result.score_f16)
    return np.asarray(scores, dtype="<u2")


def rtl_rope(q_bits: np.ndarray) -> np.ndarray:
    require(q_bits.shape == (QUERY_HEADS * HEAD_DIM,), "Q geometry mismatch")
    result = np.empty_like(q_bits)
    for head in range(QUERY_HEADS):
        base = head * HEAD_DIM
        for pair in range(HEAD_DIM // 2):
            cosine, sine = qwen2_coefficient(POSITION, pair)
            low, high, invalid, saturation = rotate_pair(
                int(q_bits[base + pair]),
                int(q_bits[base + pair + HEAD_DIM // 2]),
                cosine,
                sine,
            )
            require(not invalid and not saturation, "RTL RoPE evaluation failed")
            result[base + pair] = low
            result[base + pair + HEAD_DIM // 2] = high
    return result


def reference_rope(q_bits: np.ndarray) -> np.ndarray:
    values = torch.from_numpy(
        q_bits.view("<f2").astype(np.float64).reshape(1, QUERY_HEADS, HEAD_DIM)
    )
    frequencies = 1.0 / (
        1_000_000.0
        ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
    )
    angles = torch.outer(
        torch.tensor([float(POSITION)], dtype=torch.float64),
        frequencies,
    )
    cosine, sine = torch.cos(angles), torch.sin(angles)
    low = values[..., : HEAD_DIM // 2]
    high = values[..., HEAD_DIM // 2 :]
    result = torch.cat(
        (
            low * cosine[:, None, :] - high * sine[:, None, :],
            high * cosine[:, None, :] + low * sine[:, None, :],
        ),
        dim=-1,
    )
    return (
        result.to(torch.float16)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .view("<u2")
    )


def ordered_f16(bits: np.ndarray) -> np.ndarray:
    values = bits.astype(np.int64)
    return np.where(
        values & 0x8000,
        0x8000 - (values & 0x7FFF),
        0x8000 + values,
    )


def comparison(actual_bits: np.ndarray, expected_bits: np.ndarray) -> dict[str, Any]:
    require(actual_bits.shape == expected_bits.shape, "comparison shape mismatch")
    actual = actual_bits.view("<f2").astype(np.float64)
    expected = expected_bits.view("<f2").astype(np.float64)
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(expected), 2.0**-14)
    ulp = np.abs(ordered_f16(actual_bits) - ordered_f16(expected_bits))
    passed = (absolute <= ABSOLUTE_TOLERANCE) | (
        (relative < RELATIVE_TOLERANCE) & (ulp <= MAX_ULP_DISTANCE)
    )
    failed = np.flatnonzero(~passed)
    first = None
    if failed.size:
        index = int(failed[0])
        first = {
            "element_index": index,
            "head": index // (POSITION + 1),
            "key_position": index % (POSITION + 1),
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
        "failure_count": int(failed.size),
        "within_tolerance": not failed.size,
        "max_absolute_error": float(absolute.max()),
        "max_relative_error": float(relative.max()),
        "max_ulp_distance": int(ulp.max()),
        "first_failure": first,
    }


def generate(attempt: Path, reference: Path, output: Path) -> dict[str, Any]:
    layer_dir = attempt / "layer01"
    comparisons = {
        position: (
            reference
            / f"rtl_comparison_layer01_position{position:03d}.json"
        )
        for position in range(POSITION + 1)
    }
    traces = {
        0: layer_dir / "position000/raw/trace.hex",
        1: layer_dir / "position001/raw/trace.hex",
        2: layer_dir / "raw/trace.hex",
    }
    payloads = {
        position: authenticated_trace(comparisons[position], traces[position])
        for position in range(POSITION + 1)
    }
    actual_norm = trace_stage(payloads[POSITION], POSITION, 0)
    actual_q = trace_stage(payloads[POSITION], POSITION, 4)
    actual_q_projection = trace_stage(payloads[POSITION], POSITION, 1)
    actual_k = np.stack(
        [
            trace_stage(payloads[position], position, 6).reshape(
                KEY_VALUE_HEADS, HEAD_DIM
            )
            for position in range(POSITION + 1)
        ]
    )
    actual_scores = trace_stage(payloads[POSITION], POSITION, 8)

    layer_record_path = reference / "layer01_generation1.json"
    layer_record = load_json(layer_record_path)
    reference_norm, reference_norm_record = reference_stage(layer_record, 0)
    reference_q_projection, reference_q_projection_record = reference_stage(
        layer_record, 1
    )
    reference_q, reference_q_record = reference_stage(layer_record, 4)
    reference_scores, reference_score_record = reference_stage(layer_record, 8)
    reference_k_record = layer_record["own_cache"]["k"]
    reference_k = authenticated_array(reference_k_record)

    combinations = {
        "rtl_q_rtl_k": score_matrix(actual_q, actual_k),
        "reference_q_rtl_k": score_matrix(reference_q, actual_k),
        "rtl_q_reference_k": score_matrix(actual_q, reference_k),
        "reference_q_reference_k": score_matrix(reference_q, reference_k),
    }
    require(
        np.array_equal(combinations["rtl_q_rtl_k"], actual_scores),
        "RTL Q/K recomputation does not reproduce preserved stage08",
    )
    require(
        np.array_equal(
            combinations["reference_q_reference_k"],
            reference_scores,
        ),
        "reference Q/K recomputation does not reproduce accepted stage08",
    )
    q_boundary_combinations = {
        "rtl_projection_rtl_rope": rtl_rope(actual_q_projection),
        "reference_projection_rtl_rope": rtl_rope(reference_q_projection),
        "rtl_projection_reference_rope": reference_rope(actual_q_projection),
        "reference_projection_reference_rope": reference_rope(
            reference_q_projection
        ),
    }
    require(
        np.array_equal(
            q_boundary_combinations["rtl_projection_rtl_rope"],
            actual_q,
        ),
        "RTL RoPE recomputation does not reproduce preserved stage04",
    )
    require(
        np.array_equal(
            q_boundary_combinations["reference_projection_reference_rope"],
            reference_q,
        ),
        "reference RoPE recomputation does not reproduce accepted stage04",
    )
    q_boundary_scores = {
        name: {
            **comparison(score_matrix(bits, actual_k), reference_scores),
            "q_rope_comparison": comparison(bits, reference_q),
        }
        for name, bits in q_boundary_combinations.items()
    }
    tensor_dir = layer_dir / "vectors/tensors"
    controlled_q_projection = binary64_q_projection(actual_norm, tensor_dir)
    reproduced_reference_q_projection = binary64_q_projection(
        reference_norm,
        tensor_dir,
    )
    require(
        np.array_equal(
            reproduced_reference_q_projection,
            reference_q_projection,
        ),
        "binary64 Q projection does not reproduce accepted stage01",
    )
    controlled_q = rtl_rope(controlled_q_projection)
    controlled_projection = {
        "rtl_norm_to_binary64_q_projection": comparison(
            controlled_q_projection,
            reference_q_projection,
        ),
        "rtl_norm_to_binary64_q_rope": comparison(
            controlled_q,
            reference_q,
        ),
        "rtl_norm_to_binary64_q_scores": comparison(
            score_matrix(controlled_q, actual_k),
            reference_scores,
        ),
        "reference_norm_reproduces_reference_q_projection": True,
    }
    results = {
        name: {
            **comparison(bits, reference_scores),
            "semantic_sha256": sha256_bytes(bits.tobytes()),
        }
        for name, bits in combinations.items()
    }
    q_repaired = results["reference_q_rtl_k"]["within_tolerance"]
    k_repaired = results["rtl_q_reference_k"]["within_tolerance"]
    projection_repaired = q_boundary_scores[
        "reference_projection_rtl_rope"
    ]["within_tolerance"]
    rope_repaired = q_boundary_scores[
        "rtl_projection_reference_rope"
    ]["within_tolerance"]
    if (
        q_repaired
        and not k_repaired
        and projection_repaired
        and not rope_repaired
        and not controlled_projection[
            "rtl_norm_to_binary64_q_scores"
        ]["within_tolerance"]
    ):
        repair = (
            "localize and repair the upstream layer00-to-layer01 hidden-state "
            "trajectory; changing only layer01 Q projection arithmetic cannot "
            "remove the material stage08 mismatch"
        )
    elif q_repaired and not k_repaired:
        repair = "align the layer01 position002 Q projection/RoPE boundary"
    elif k_repaired and not q_repaired:
        repair = "align the layer01 prefix K projection/RoPE cache boundary"
    elif q_repaired and k_repaired:
        repair = (
            "align either layer01 position002 Q or prefix K projection/RoPE "
            "boundary; each independently removes the material stage08 failure"
        )
    else:
        repair = (
            "align both layer01 position002 Q and prefix K projection/RoPE "
            "boundaries; neither isolated replacement removes the failure"
        )

    first_index = 30
    first_values = {}
    for name, bits in combinations.items():
        first_values[name] = {
            "bits": f"{int(bits[first_index]):04x}",
            "value": float(bits[first_index : first_index + 1].view("<f2")[0]),
        }
    first_values["preserved_rtl_stage08"] = {
        "bits": f"{int(actual_scores[first_index]):04x}",
        "value": float(
            actual_scores[first_index : first_index + 1].view("<f2")[0]
        ),
    }
    first_values["accepted_reference_stage08"] = {
        "bits": f"{int(reference_scores[first_index]):04x}",
        "value": float(
            reference_scores[first_index : first_index + 1].view("<f2")[0]
        ),
    }

    output.mkdir(parents=True, exist_ok=False)
    document = {
        "schema_version": 1,
        "kind": "ace3_layer01_position002_stage08_controlled_comparison",
        "status": "ROOT_CAUSE_REFINED_UPSTREAM",
        "policy": {
            "profile": "native AWQ W4A16 G128 asymmetric, FP16 activations/KV",
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance_strict": RELATIVE_TOLERANCE,
            "max_ulp_distance": MAX_ULP_DISTANCE,
        },
        "boundary": {
            "layer": LAYER,
            "position": POSITION,
            "stage": 8,
            "stage_name": "attention_qk",
            "first_failure_head": 10,
            "first_failure_key_position": 0,
        },
        "authenticated_inputs": {
            "attempt_traces": [
                file_record(traces[position])
                for position in range(POSITION + 1)
            ],
            "rtl_comparisons": [
                file_record(comparisons[position])
                for position in range(POSITION + 1)
            ],
            "reference_layer_record": file_record(layer_record_path),
            "reference_norm": reference_norm_record,
            "reference_q_projection": reference_q_projection_record,
            "reference_q": reference_q_record,
            "reference_k": reference_k_record,
            "reference_scores": reference_score_record,
            "q_projection_tensors": [
                file_record(path)
                for path in sorted(
                    tensor_dir.glob("layer1_self_attn_q_proj_*.hex")
                )
            ],
            "diagnostic_source": file_record(Path(__file__)),
        },
        "input_comparisons": {
            "position002_input_rmsnorm": comparison(
                actual_norm,
                reference_norm,
            ),
            "position002_q_rope": comparison(actual_q, reference_q),
            "prefix_and_position002_k_rope": comparison(
                actual_k.reshape(-1),
                reference_k.reshape(-1),
            ),
        },
        "score_combinations": results,
        "q_projection_rope_decomposition": q_boundary_scores,
        "controlled_projection_from_actual_norm": controlled_projection,
        "first_reported_failure_values": first_values,
        "root_cause": (
            "stage08 arithmetic and trace ordering are correct: exact "
            "recomputation reproduces both preserved RTL and accepted-reference "
            "scores. The prior crossed-Q result replaced both the Q-projection "
            "arithmetic and its free-running operand. A controlled binary64 "
            "Q projection fed from the preserved RTL stage00 still has all "
            "material stage08 failures, while the same implementation exactly "
            "reproduces accepted stage01 from accepted stage00. The divergence "
            "therefore enters through the upstream layer00-to-layer01 hidden "
            "trajectory rather than layer01 Q-projection arithmetic; cached K "
            "and RTL RoPE remain negative controls."
        ),
        "repair_required": repair,
        "claim_boundary": (
            "bounded layer01 numerical diagnosis only; no full traversal, "
            "synthesis, PPA, FPGA, or dialogue claim"
        ),
    }
    result_path = output / "diagnosis.json"
    result_path.write_bytes(canonical_json(document))
    print(
        "LAYER01_POSITION002_STAGE08_ROOT_CAUSE_REFINED_UPSTREAM "
        f"result={result_path} repair={repair}"
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(
        args.attempt.resolve(strict=True),
        args.reference.resolve(strict=True),
        args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
