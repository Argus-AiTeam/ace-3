#!/usr/bin/env python3
"""Build and validate fresh token-271 position-2 RTL traversal evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors import safe_open

import official_model24_dialogue as dialogue_oracle
from decoder_layer0_oracle import run_token as run_decoder_layer_token
from fp16_adaptation_oracle import silu_gate_exp
from model24_oracle import authenticate_checkpoint
from official_model24_dialogue import (
    _load_model,
    _reference_layer_step,
)
from official_model24_next_token import _layer_tensor_names, _tensor_record
from qwen2_rope_oracle import qwen2_coefficient

LAYER_COUNT = 24
HIDDEN_SIZE = 896
POSITION0_TOKEN_ID = 151644
POSITION1_TOKEN_ID = 2114
SELECTED_TOKEN_ID = 271
POSITION = 2
CHECKPOINT_SHA256 = (
    "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
)
EMBEDDING_SHA256 = (
    "50b4ddfa1a96b344436c1c099374abeb4477c7b7cb6ac1200d6a64e68b8e1edb"
)
POSITION0_EMBEDDING_SHA256 = (
    "4debe5d7a410d8208c773ef054454236a08ccc89710c0c06186853119815428c"
)
POSITION1_EMBEDDING_SHA256 = (
    "7c6cad8e630f773207e90bcb4606c096585e5ca27a873d0089fe575089bd627f"
)
TOLERANCE = 0.125

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "build/model24_rtl_cascade/checkpoint/model.safetensors"
DEFAULT_OUTPUT = ROOT / "build/model24_selected_token_position2/evidence.json"
CONSUMED_SOURCE_PATHS = {
    "validator": "ace3/model/validate_selected_token_position2_traversal.py",
    "kv_import_and_independent_oracle": "ace3/model/official_model24_dialogue.py",
    "official_model24_next_token": "ace3/model/official_model24_next_token.py",
    "official_single_decoder_layer": "ace3/model/official_single_decoder_layer.py",
    "model24_execution_oracle": "ace3/model/model24_execution_oracle.py",
    "model24_oracle": "ace3/model/model24_oracle.py",
    "decoder_layer0_oracle": "ace3/model/decoder_layer0_oracle.py",
    "attention_oracle": "ace3/model/attention_oracle.py",
    "fp16_adaptation_oracle": "ace3/model/fp16_adaptation_oracle.py",
    "awq_bit_oracle": "ace3/model/awq_bit_oracle.py",
    "projection_oracle": "ace3/model/projection_oracle.py",
    "qwen2_rope_oracle": "ace3/model/qwen2_rope_oracle.py",
    "fp16_fixed_rtl": "ace3/rtl/ace3_fp16_fixed.sv",
    "projection_rounder_rtl": "ace3/rtl/ace3_q47_48_to_f16_rne.sv",
    "awq_dot_lane_rtl": "ace3/rtl/ace3_awq_w4a16_g128_dot_lane.sv",
    "projection_rtl": "ace3/rtl/ace3_awq_w4a16_projection_engine.sv",
    "fp16_rms_rtl": "ace3/rtl/ace3_fp16_rmsnorm_core.sv",
    "fp16_residual_rtl": "ace3/rtl/ace3_fp16_residual_add_core.sv",
    "decoder_rtl": "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
    "fp16_silu_rtl": "ace3/rtl/ace3_fp16_silu_gate_core.sv",
    "qwen2_rope_rtl": "ace3/rtl/ace3_qwen2_rope_pair.sv",
    "fp16_kv_cache_rtl": "ace3/rtl/ace3_fp16_kv_cache.sv",
    "attention_score_rtl": "ace3/rtl/ace3_attention_score_core.sv",
    "attention_softmax_rtl": "ace3/rtl/ace3_attention_softmax_core.sv",
    "attention_value_rtl": "ace3/rtl/ace3_attention_value_core.sv",
    "decoder_qzeros_address_rtl": "ace3/rtl/ace3_decoder_qzeros_address.sv",
    "decoder_verilator_harness": (
        "ace3/tb/ace3_decoder_layer0_token_engine_main.cpp"
    ),
    "trace_capture_policy": "ace3/tb/ace3_layer0_trace_capture_policy.h",
    "makefile_target": "Makefile",
    "focused_tests": (
        "ace3/model/tests/test_validate_selected_token_position2_traversal.py"
    ),
}


class ContinuationError(RuntimeError):
    """Raised when durable continuation evidence is incomplete or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContinuationError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while payload := stream.read(1024 * 1024):
            digest.update(payload)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(document, dict), f"{path} must contain a JSON object")
    return document


def canonical_json(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def selected_accurate_silu_gate(
    gate_f16: int,
    up_f16: int,
) -> tuple[int, bool, bool]:
    return silu_gate_exp(gate_f16, up_f16)


def selected_primary_layer_step(
    state: Any,
    hidden: np.ndarray,
    start_position: int,
) -> np.ndarray:
    previous_silu_gate = dialogue_oracle._fp16_silu_gate
    dialogue_oracle._fp16_silu_gate = selected_accurate_silu_gate
    try:
        return dialogue_oracle._primary_layer_step(
            state,
            hidden,
            start_position,
        )
    finally:
        dialogue_oracle._fp16_silu_gate = previous_silu_gate


def semantic_hidden_sha256(path: Path, expected_rows: int = HIDDEN_SIZE) -> str:
    rows = path.read_text(encoding="ascii").splitlines()
    require(len(rows) == expected_rows, f"{path} hidden row count mismatch")
    payload = bytearray()
    for index, row in enumerate(rows):
        require(
            len(row) == 10
            and row[:2] == "00"
            and int(row[2:6], 16) == index,
            f"{path} hidden row {index} ordering mismatch",
        )
        payload.extend(int(row[6:10], 16).to_bytes(2, "little"))
    return sha256_bytes(bytes(payload))


def require_ordered_layers(layers: Sequence[Mapping[str, Any]]) -> None:
    require(
        [layer.get("layer_index") for layer in layers] == list(range(LAYER_COUNT)),
        "layer records are not ordered 0 through 23",
    )


def content_matches(path: Path, record: Mapping[str, Any], label: str) -> dict[str, Any]:
    actual = file_record(path)
    require(
        actual["bytes"] == record.get("bytes")
        and actual["sha256"] == record.get("sha256"),
        f"{label} content binding mismatch",
    )
    return actual


def consumed_source_records(
    repository_root: Path = ROOT,
) -> dict[str, dict[str, Any]]:
    return {
        label: file_record(repository_root / relative_path)
        for label, relative_path in CONSUMED_SOURCE_PATHS.items()
    }


def require_consumed_source_bindings(
    stored: Mapping[str, Any],
    repository_root: Path = ROOT,
) -> None:
    actual = consumed_source_records(repository_root)
    require(
        set(actual).issubset(stored),
        "consumed source closure is incomplete",
    )
    for label, record in actual.items():
        require(
            stored.get(label) == record,
            f"{label} source binding mismatch",
        )


def official_embedding_binding() -> dict[str, Any]:
    authenticate_checkpoint(CHECKPOINT)
    checkpoint_record = file_record(CHECKPOINT)
    require(
        checkpoint_record["sha256"] == CHECKPOINT_SHA256,
        "official checkpoint SHA-256 mismatch",
    )
    with safe_open(CHECKPOINT, framework="np") as checkpoint:
        weights = np.asarray(
            checkpoint.get_tensor("model.embed_tokens.weight"),
            dtype="<f2",
        )
        require(
            weights.shape == (151936, HIDDEN_SIZE),
            "official tied embedding geometry mismatch",
        )
        embedding_bits = np.asarray(
            weights[SELECTED_TOKEN_ID],
            dtype="<f2",
        ).view("<u2")
    digest = sha256_bytes(embedding_bits.tobytes())
    require(digest == EMBEDDING_SHA256, "token-271 embedding SHA-256 mismatch")
    return {
        "checkpoint": checkpoint_record,
        "tensor": "model.embed_tokens.weight",
        "token_id": SELECTED_TOKEN_ID,
        "dtype": "FP16",
        "elements": HIDDEN_SIZE,
        "sha256": digest,
    }


def token_embedding_bits(
    token_id: int,
    expected_sha256: str,
    label: str,
) -> np.ndarray:
    with safe_open(CHECKPOINT, framework="np") as checkpoint:
        embedding = np.asarray(
            checkpoint.get_tensor("model.embed_tokens.weight")[token_id],
            dtype="<f2",
        ).view("<u2")
    require(
        sha256_bytes(embedding.tobytes()) == expected_sha256,
        f"{label} embedding SHA-256 mismatch",
    )
    return embedding


def selected_embedding_bits() -> np.ndarray:
    return token_embedding_bits(
        SELECTED_TOKEN_ID,
        EMBEDDING_SHA256,
        "token-271",
    )


def hidden_payload(bits: np.ndarray) -> bytes:
    values = np.asarray(bits, dtype="<u2")
    require(values.shape == (HIDDEN_SIZE,), "hidden activation geometry mismatch")
    return "".join(
        f"00{index:04x}{int(value):04x}\n"
        for index, value in enumerate(values)
    ).encode("ascii")


def load_hidden_bits(path: Path) -> np.ndarray:
    rows = path.read_text(encoding="ascii").splitlines()
    require(len(rows) == HIDDEN_SIZE, f"{path} hidden row count mismatch")
    values = np.empty(HIDDEN_SIZE, dtype="<u2")
    for index, row in enumerate(rows):
        require(
            len(row) == 10
            and row[:2] == "00"
            and int(row[2:6], 16) == index,
            f"{path} hidden row {index} ordering mismatch",
        )
        values[index] = int(row[6:10], 16)
    return values


def exact_hex_comparison(
    actual_payload: bytes,
    expected_payload: bytes,
    artifact: str,
) -> dict[str, Any]:
    require(artifact in {"trace", "final_hidden"}, "unsupported exact artifact")
    actual_rows = actual_payload.splitlines()
    expected_rows = expected_payload.splitlines()
    mismatch_ordinals = [
        ordinal
        for ordinal in range(max(len(actual_rows), len(expected_rows)))
        if (
            ordinal >= len(actual_rows)
            or ordinal >= len(expected_rows)
            or actual_rows[ordinal] != expected_rows[ordinal]
        )
    ]
    first_difference = None
    if mismatch_ordinals:
        ordinal = mismatch_ordinals[0]
        actual = actual_rows[ordinal] if ordinal < len(actual_rows) else None
        expected = expected_rows[ordinal] if ordinal < len(expected_rows) else None
        first_difference = {
            "row": ordinal,
            "actual": None if actual is None else actual.decode("ascii"),
            "expected": None if expected is None else expected.decode("ascii"),
        }
        selected = actual if actual is not None else expected
        assert selected is not None
        if artifact == "trace" and len(selected) == 16:
            first_difference.update(
                {
                    "signal": (
                        f"trace[position={int(selected[2:6], 16)},"
                        f"stage={int(selected[6:8], 16)},"
                        f"index={int(selected[8:12], 16)}]"
                    ),
                    "actual_bits": (
                        None if actual is None else f"0x{int(actual[12:16], 16):04x}"
                    ),
                    "expected_bits": (
                        None
                        if expected is None
                        else f"0x{int(expected[12:16], 16):04x}"
                    ),
                }
            )
        elif artifact == "final_hidden" and len(selected) == 10:
            first_difference.update(
                {
                    "signal": f"final_hidden[{int(selected[2:6], 16)}]",
                    "actual_bits": (
                        None if actual is None else f"0x{int(actual[6:10], 16):04x}"
                    ),
                    "expected_bits": (
                        None
                        if expected is None
                        else f"0x{int(expected[6:10], 16):04x}"
                    ),
                }
            )
    return {
        "artifact": artifact,
        "actual_rows": len(actual_rows),
        "expected_rows": len(expected_rows),
        "actual_sha256": sha256_bytes(actual_payload),
        "expected_sha256": sha256_bytes(expected_payload),
        "mismatch_count": len(mismatch_ordinals),
        "exact_match": not mismatch_ordinals,
        "first_difference": first_difference,
    }


def layer_oracle_values(
    checkpoint: Any,
    layer_index: int,
    vectors: Mapping[str, Any],
) -> dict[str, list[int]]:
    prefix = f"model.layers.{layer_index}."
    vector_tensors = {
        tensor["checkpoint_tensor"]["name"]: tensor["checkpoint_tensor"]
        for tensor in vectors["tensors"]
    }
    values: dict[str, list[int]] = {}
    for name in _layer_tensor_names(layer_index):
        value = np.ascontiguousarray(checkpoint.get_tensor(name))
        require(
            vector_tensors.get(name) == _tensor_record(name, value),
            f"layer {layer_index} oracle tensor/vector binding mismatch: {name}",
        )
        unit_dtype = "<u2" if value.dtype == np.dtype("<f2") else "<u4"
        values[f"model.layers.0.{name.removeprefix(prefix)}:"] = (
            value.view(unit_dtype).reshape(-1).astype(np.uint64).tolist()
        )
    return values


def oracle_trace_payload(
    trace: Sequence[tuple[int, int, int, int]],
) -> bytes:
    return "".join(
        f"00{position:04x}{stage:02x}{index:04x}{value:04x}\n"
        for stage, index, value, position in trace
    ).encode("ascii")


def parse_natural_terminal(
    path: Path,
    layer_index: int,
    position: int = POSITION,
) -> dict[str, int]:
    try:
        lines = path.read_bytes().decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ContinuationError(f"layer {layer_index} terminal is not ASCII") from error
    require(len(lines) == 1, f"layer {layer_index} terminal line count mismatch")
    fields: dict[str, str] = {}
    for field in lines[0].split():
        require(
            field.count("=") == 1,
            f"layer {layer_index} terminal field is malformed",
        )
        key, value = field.split("=", 1)
        require(key not in fields, f"layer {layer_index} terminal field is duplicated")
        fields[key] = value
    require(
        set(fields)
        == {
            "schema",
            "layer_index",
            "position",
            "natural_terminal",
            "exit_code",
            "trace_count",
            "final_count",
            "done_count",
        },
        f"layer {layer_index} terminal schema mismatch",
    )
    require(
        fields["schema"] == "ace3_decoder_token_transaction_v1"
        and fields["layer_index"] == str(layer_index)
        and fields["position"] == str(position)
        and fields["natural_terminal"] == "1"
        and fields["exit_code"] == "0",
        f"layer {layer_index} did not reach a natural terminal",
    )
    counts = {
        name: int(fields[name])
        for name in ("trace_count", "final_count", "done_count")
    }
    require(
        counts["trace_count"] > 0
        and counts["final_count"] == HIDDEN_SIZE
        and counts["done_count"] == 1,
        f"layer {layer_index} terminal count mismatch",
    )
    return counts


def run_logged(command: Sequence[str], log_path: Path) -> None:
    with log_path.open("wb") as log:
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    require(
        completed.returncode == 0,
        f"command failed with exit {completed.returncode}: {' '.join(command)}",
    )


def materialize_input_vectors(
    layer_index: int,
    position: int,
    hidden_bits: np.ndarray,
    vector_dir: Path,
) -> dict[str, Any]:
    vector_dir.mkdir(parents=True)
    input_path = vector_dir / "inputs.hex"
    input_path.write_bytes(hidden_payload(hidden_bits))
    rope_path = vector_dir / "rope_coefficients.hex"
    rope_path.write_text(
        "".join(
            f"{rope_position:04x}{pair:02x}{cosine:04x}{sine:04x}\n"
            for rope_position in range(position + 1)
            for pair in range(32)
            for cosine, sine in (qwen2_coefficient(rope_position, pair),)
        ),
        encoding="ascii",
    )
    return {
        "layer_index": layer_index,
        "position": position,
        "input": file_record(input_path),
        "rope_coefficients": file_record(rope_path),
    }


def materialize_transaction_vectors(
    checkpoint: Any,
    layer_index: int,
    position: int,
    hidden_bits: np.ndarray,
    vector_dir: Path,
) -> dict[str, Any]:
    activation_vectors = materialize_input_vectors(
        layer_index,
        position,
        hidden_bits,
        vector_dir,
    )
    tensor_dir = vector_dir / "tensors"
    tensor_dir.mkdir()
    prefix = f"model.layers.{layer_index}."
    tensors = []
    for name in _layer_tensor_names(layer_index):
        value = np.ascontiguousarray(checkpoint.get_tensor(name))
        require(
            value.dtype in (np.dtype("<f2"), np.dtype("<i4")),
            f"{name} unsupported tensor dtype",
        )
        unit = value.dtype.itemsize
        payload = value.tobytes()
        suffix = name.removeprefix(prefix).replace(".", "_")
        serialized = (
            f"layer{layer_index}_{suffix}."
            f"{'fp16le.bin' if unit == 2 else 'i32le.bin'}.hex"
        )
        path = tensor_dir / serialized
        path.write_text(
            "".join(
                f"{int.from_bytes(payload[offset:offset + unit], 'little'):0{unit * 2}x}\n"
                for offset in range(0, len(payload), unit)
            ),
            encoding="ascii",
        )
        tensors.append(
            {
                "checkpoint_tensor": _tensor_record(name, value),
                "serialized": file_record(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "ace3_position2_live_transaction_vectors",
        "layer_index": layer_index,
        "position": position,
        "input_activation_sha256": sha256_bytes(
            np.asarray(hidden_bits, dtype="<u2").tobytes()
        ),
        "input": activation_vectors["input"],
        "rope_coefficients": activation_vectors["rope_coefficients"],
        "tensors": tensors,
    }
    manifest_path = vector_dir / "manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    return {
        "manifest": file_record(manifest_path),
        "input": manifest["input"],
        "rope_coefficients": manifest["rope_coefficients"],
        "tensors": tensors,
    }


def execute_transaction(
    binary: Path,
    layer_index: int,
    position: int,
    hidden_bits: np.ndarray,
    vectors: Mapping[str, Any],
    tensor_vector_dir: Path,
    transaction_dir: Path,
    state_out: Path,
    state_in: Path | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    raw_dir = transaction_dir / "raw"
    raw_dir.mkdir(parents=True)
    simulation_log = transaction_dir / "simulation.log"
    command = [
        str(binary),
        "--layer-index",
        str(layer_index),
        "--vector-dir",
        str(Path(vectors["input"]["path"]).parent),
        "--tensor-dir",
        str(tensor_vector_dir),
        "--raw-dir",
        str(raw_dir),
        "--transaction-position",
        str(position),
        "--state-out",
        str(state_out),
        "--progress-interval",
        "1000000",
    ]
    if state_in is not None:
        command.extend(("--state-in", str(state_in)))
    run_logged(command, simulation_log)
    counts = parse_natural_terminal(
        raw_dir / "terminal.txt",
        layer_index,
        position,
    )
    output_path = raw_dir / "final.hex"
    output_sha256 = semantic_hidden_sha256(output_path)
    input_sha256 = sha256_bytes(np.asarray(hidden_bits, dtype="<u2").tobytes())
    require(
        vectors["input"]["sha256"] == sha256_bytes(hidden_payload(hidden_bits)),
        f"layer {layer_index} position {position} vector input binding mismatch",
    )
    record = {
        "layer_index": layer_index,
        "position": position,
        "input": {
            "dtype": "FP16",
            "elements": HIDDEN_SIZE,
            "sha256": input_sha256,
        },
        "output": {
            **file_record(output_path),
            "dtype": "FP16",
            "elements": HIDDEN_SIZE,
            "semantic_sha256": output_sha256,
        },
        "output_state": file_record(state_out),
        "vectors": dict(vectors),
        "simulation_log": file_record(simulation_log),
        "raw": {
            "terminal": file_record(raw_dir / "terminal.txt"),
            "trace": file_record(raw_dir / "trace.hex"),
            **counts,
        },
    }
    return record, load_hidden_bits(output_path)


def compare_live_layers(
    layers: Sequence[Mapping[str, Any]],
    embedding_bits: np.ndarray,
    *,
    write_reports: bool,
) -> list[dict[str, Any]]:
    torch.set_num_threads(1)
    _, _, _, states = _load_model(CHECKPOINT)
    compared = []
    previous_bits = np.asarray(embedding_bits, dtype="<u2")
    position0_bits = token_embedding_bits(
        POSITION0_TOKEN_ID,
        POSITION0_EMBEDDING_SHA256,
        "position-0 token",
    )
    position1_bits = token_embedding_bits(
        POSITION1_TOKEN_ID,
        POSITION1_EMBEDDING_SHA256,
        "position-1 token",
    )
    for layer_index, (layer, state) in enumerate(zip(layers, states, strict=True)):
        require(
            layer.get("layer_index") == layer_index,
            f"layer {layer_index} comparison ordering mismatch",
        )
        replay = layer.get("fresh_prefix", {})
        actual_position0 = load_hidden_bits(
            Path(replay["position0"]["output"]["path"])
        )
        actual_position1 = load_hidden_bits(
            Path(replay["position1"]["output"]["path"])
        )
        expected_position0 = selected_primary_layer_step(
            state,
            position0_bits.reshape(1, HIDDEN_SIZE),
            0,
        )[0]
        reference_position0 = _reference_layer_step(
            state,
            torch.from_numpy(
                position0_bits.view("<f2").astype(np.float64).reshape(1, HIDDEN_SIZE)
            ),
            0,
        )[0].detach().cpu().numpy()
        require(
            np.array_equal(actual_position0, expected_position0),
            f"layer {layer_index} position-0 replay differs from exact integer oracle",
        )
        expected_position1 = selected_primary_layer_step(
            state,
            position1_bits.reshape(1, HIDDEN_SIZE),
            1,
        )[0]
        reference_position1 = _reference_layer_step(
            state,
            torch.from_numpy(
                position1_bits.view("<f2").astype(np.float64).reshape(1, HIDDEN_SIZE)
            ),
            1,
        )[0].detach().cpu().numpy()
        require(
            np.array_equal(actual_position1, expected_position1),
            f"layer {layer_index} position-1 replay differs from exact integer oracle",
        )
        parent_k = np.asarray(state.primary_k[:2], dtype="<u2").copy()
        parent_v = np.asarray(state.primary_v[:2], dtype="<u2").copy()
        actual_bits = load_hidden_bits(Path(layer["output"]["path"]))
        primary = selected_primary_layer_step(
            state,
            previous_bits.reshape(1, HIDDEN_SIZE),
            POSITION,
        )[0]
        reference = _reference_layer_step(
            state,
            torch.from_numpy(
                previous_bits.view("<f2").astype(np.float64).reshape(1, HIDDEN_SIZE)
            ),
            POSITION,
        )[0].detach().cpu().numpy()
        require(
            np.array_equal(actual_bits, primary),
            f"layer {layer_index} RTL output differs from exact integer oracle",
        )
        actual_values = actual_bits.view("<f2").astype(np.float64)
        difference = np.abs(actual_values - reference)
        max_error = float(difference.max())
        comparison = {
            "implementation": (
                "independent ACE-3 integer W4A16 oracle with selected exp "
                "range-reduced degree-7 Q24 SiLU plus PyTorch CPU binary64 "
                "dequantized-AWQ reference"
            ),
            "seed": "authenticated token-271 tied embedding",
            "inter_layer_boundary": "binary16 round after every RTL layer",
            "fresh_prefix": {
                "positions": [0, 1],
                "position0_exact_output_sha256": sha256_bytes(
                    np.asarray(expected_position0, dtype="<u2").tobytes()
                ),
                "position1_exact_output_sha256": sha256_bytes(
                    np.asarray(expected_position1, dtype="<u2").tobytes()
                ),
                "position0_reference_max_abs_error": float(
                    np.abs(
                        actual_position0.view("<f2").astype(np.float64)
                        - reference_position0
                    ).max()
                ),
                "position1_reference_max_abs_error": float(
                    np.abs(
                        actual_position1.view("<f2").astype(np.float64)
                        - reference_position1
                    ).max()
                ),
                "exact_integer_oracle_match": True,
            },
            "exact_integer_oracle_output_sha256": sha256_bytes(
                np.asarray(primary, dtype="<u2").tobytes()
            ),
            "rtl_matches_exact_integer_oracle": True,
            "max_abs_error": max_error,
            "mean_abs_error": float(difference.mean()),
            "absolute_tolerance": TOLERANCE,
            "within_tolerance": max_error <= TOLERANCE,
        }
        require(
            comparison["within_tolerance"],
            f"layer {layer_index} independent comparison exceeds tolerance",
        )
        report_path = Path(layer["output"]["path"]).parent / "comparison.json"
        if write_reports:
            report_path.write_bytes(canonical_json(comparison))
        else:
            require(
                load_json(report_path) == comparison,
                f"layer {layer_index} comparison report mismatch",
            )
        base = {
            key: value
            for key, value in layer.items()
            if key
            not in {
                "fp16_kv_parentage",
                "independent_oracle_comparison",
                "comparison_report",
            }
        }
        compared.append(
            {
                **base,
                "fp16_kv_parentage": {
                    "schema_version": 1,
                    "kind": "ace3_current_worktree_position01_exact_replay",
                    "layer_id": layer_index,
                    "source_positions": [0, 1],
                    "target_position": POSITION,
                    "k_tensor_sha256": sha256_bytes(parent_k.tobytes()),
                    "v_tensor_sha256": sha256_bytes(parent_v.tobytes()),
                    "position0_trace": replay["position0"]["raw"]["trace"],
                    "position1_trace": replay["position1"]["raw"]["trace"],
                    "position2_state": replay["position1"]["output_state"],
                    "exact_integer_oracle_match": True,
                },
                "independent_oracle_comparison": comparison,
                "comparison_report": file_record(report_path),
            }
        )
        position0_bits = actual_position0
        position1_bits = actual_position1
        previous_bits = actual_bits
    return compared


def execute_live_traversal(
    live_root: Path,
    embedding_bits: np.ndarray,
) -> dict[str, Any]:
    require(not live_root.exists(), "live traversal output already exists")
    live_root.mkdir(parents=True)
    layers = []
    hidden_bits = np.asarray(embedding_bits, dtype="<u2")
    position0_bits = token_embedding_bits(
        POSITION0_TOKEN_ID,
        POSITION0_EMBEDDING_SHA256,
        "position-0 token",
    )
    position1_bits = token_embedding_bits(
        POSITION1_TOKEN_ID,
        POSITION1_EMBEDDING_SHA256,
        "position-1 token",
    )
    with safe_open(CHECKPOINT, framework="np") as checkpoint:
        for layer_index in range(LAYER_COUNT):
            layer_dir = live_root / f"layer{layer_index:02d}"
            vector_dir = layer_dir / "vectors"
            binary = (
                live_root
                / f"compiled/layer{layer_index}/obj_dir"
                / "Vace3_decoder_layer0_token_engine"
            )
            vectors = materialize_transaction_vectors(
                checkpoint,
                layer_index,
                POSITION,
                hidden_bits,
                vector_dir,
            )
            position0_vectors = materialize_input_vectors(
                layer_index,
                0,
                position0_bits,
                layer_dir / "position000/vectors",
            )
            position1_vectors = materialize_input_vectors(
                layer_index,
                1,
                position1_bits,
                layer_dir / "position001/vectors",
            )
            compile_log = layer_dir / "compile.log"
            run_logged(
                [
                    "make",
                    "--no-print-directory",
                    "model24-rtl-layer-compile",
                    f"MODEL24_RTL_LAYER_INDEX={layer_index}",
                    "MODEL24_RTL_ACCURATE_SILU=1",
                    f"MODEL24_RTL_CASCADE_DIR={live_root}",
                ],
                compile_log,
            )
            require(binary.is_file(), f"layer {layer_index} live RTL binary is missing")
            oracle_values = layer_oracle_values(checkpoint, layer_index, vectors)
            cache_k: list[list[int]] = []
            cache_v: list[list[int]] = []
            position0, position0_bits = execute_exact_transaction(
                binary,
                layer_index,
                0,
                position0_bits,
                position0_vectors,
                vector_dir,
                layer_dir / "position000",
                layer_dir / "position001.state",
                oracle_values,
                cache_k,
                cache_v,
            )
            position1, position1_bits = execute_exact_transaction(
                binary,
                layer_index,
                1,
                position1_bits,
                position1_vectors,
                vector_dir,
                layer_dir / "position001",
                layer_dir / "position002.state",
                oracle_values,
                cache_k,
                cache_v,
                Path(position0["output_state"]["path"]),
            )
            current, hidden_bits = execute_exact_transaction(
                binary,
                layer_index,
                POSITION,
                hidden_bits,
                vectors,
                vector_dir,
                layer_dir,
                layer_dir / "position003.state",
                oracle_values,
                cache_k,
                cache_v,
                Path(position1["output_state"]["path"]),
            )
            layers.append(
                {
                    "layer_index": layer_index,
                    "position": POSITION,
                    "input": current["input"],
                    "output": current["output"],
                    "position01_state": position1["output_state"],
                    "fresh_prefix": {
                        "position0": position0,
                        "position1": position1,
                    },
                    "output_state": current["output_state"],
                    "vectors": vectors,
                    "live_binary": file_record(binary),
                    "compile_log": file_record(compile_log),
                    "simulation_log": current["simulation_log"],
                    "raw": current["raw"],
                    "exact_oracle": current["exact_oracle"],
                    "exact_comparison": current["exact_comparison"],
                }
            )
    layers = compare_live_layers(layers, embedding_bits, write_reports=True)
    return {
        "status": "COMPLETE",
        "execution": "current-worktree compiled Verilator RTL",
        "selected_token_id": SELECTED_TOKEN_ID,
        "position": POSITION,
        "layer_order": list(range(LAYER_COUNT)),
        "natural_terminal_layers": LAYER_COUNT,
        "fresh_token_inputs": {
            "position0": {
                "token_id": POSITION0_TOKEN_ID,
                "embedding_sha256": POSITION0_EMBEDDING_SHA256,
            },
            "position1": {
                "token_id": POSITION1_TOKEN_ID,
                "embedding_sha256": POSITION1_EMBEDDING_SHA256,
            },
        },
        "layers": layers,
        "post_layer23": {
            "hidden_sha256": layers[-1]["output"]["semantic_sha256"],
            "natural_terminal": True,
            "independent_oracle_within_tolerance": True,
        },
    }


def validate_live_traversal(
    traversal: Mapping[str, Any],
    embedding_bits: np.ndarray,
) -> dict[str, Any]:
    require(
        traversal.get("status") == "COMPLETE"
        and traversal.get("execution") == "current-worktree compiled Verilator RTL"
        and traversal.get("selected_token_id") == SELECTED_TOKEN_ID
        and traversal.get("position") == POSITION
        and traversal.get("layer_order") == list(range(LAYER_COUNT))
        and traversal.get("natural_terminal_layers") == LAYER_COUNT,
        "live traversal identity mismatch",
    )
    require(
        traversal.get("fresh_token_inputs")
        == {
            "position0": {
                "token_id": POSITION0_TOKEN_ID,
                "embedding_sha256": POSITION0_EMBEDDING_SHA256,
            },
            "position1": {
                "token_id": POSITION1_TOKEN_ID,
                "embedding_sha256": POSITION1_EMBEDDING_SHA256,
            },
        },
        "fresh token input binding mismatch",
    )
    layers = traversal.get("layers")
    require(isinstance(layers, list), "live traversal layers are missing")
    require_ordered_layers(layers)
    previous_sha256 = sha256_bytes(np.asarray(embedding_bits, dtype="<u2").tobytes())
    position0_sha256 = POSITION0_EMBEDDING_SHA256
    position1_sha256 = POSITION1_EMBEDDING_SHA256
    for layer_index, layer in enumerate(layers):
        require(
            layer.get("input", {}).get("sha256") == previous_sha256,
            f"layer {layer_index} activation lineage mismatch",
        )
        replay = layer.get("fresh_prefix", {})
        for replay_label, replay_position, expected_input in (
            ("position0", 0, position0_sha256),
            ("position1", 1, position1_sha256),
        ):
            transaction = replay.get(replay_label, {})
            require(
                transaction.get("layer_index") == layer_index
                and transaction.get("position") == replay_position
                and transaction.get("input", {}).get("sha256") == expected_input,
                f"layer {layer_index} {replay_label} replay identity mismatch",
            )
            for label in ("output", "output_state", "simulation_log"):
                content_matches(
                    Path(transaction[label]["path"]),
                    transaction[label],
                    f"layer {layer_index} {replay_label} {label}",
                )
            replay_vectors = transaction.get("vectors", {})
            for label in ("input", "rope_coefficients"):
                content_matches(
                    Path(replay_vectors[label]["path"]),
                    replay_vectors[label],
                    f"layer {layer_index} {replay_label} vector {label}",
                )
            require(
                semantic_hidden_sha256(Path(replay_vectors["input"]["path"]))
                == expected_input,
                f"layer {layer_index} {replay_label} vector semantics mismatch",
            )
            replay_raw = transaction.get("raw", {})
            for label in ("terminal", "trace"):
                content_matches(
                    Path(replay_raw[label]["path"]),
                    replay_raw[label],
                    f"layer {layer_index} {replay_label} raw {label}",
                )
            counts = parse_natural_terminal(
                Path(replay_raw["terminal"]["path"]),
                layer_index,
                replay_position,
            )
            require(
                all(replay_raw.get(name) == value for name, value in counts.items()),
                f"layer {layer_index} {replay_label} terminal count mismatch",
            )
            replay_output_sha256 = semantic_hidden_sha256(
                Path(transaction["output"]["path"])
            )
            require(
                transaction["output"].get("semantic_sha256")
                == replay_output_sha256,
                f"layer {layer_index} {replay_label} output binding mismatch",
            )
            validate_exact_transaction_artifacts(
                transaction,
                layer_index,
                replay_position,
            )
            if replay_position == 0:
                position0_sha256 = replay_output_sha256
            else:
                position1_sha256 = replay_output_sha256
        require(
            layer.get("position01_state")
            == replay["position1"]["output_state"],
            f"layer {layer_index} fresh position-0/1 state mismatch",
        )
        for label in (
            "output",
            "output_state",
            "live_binary",
            "compile_log",
            "simulation_log",
            "comparison_report",
        ):
            content_matches(
                Path(layer[label]["path"]),
                layer[label],
                f"layer {layer_index} {label}",
            )
        vectors = layer.get("vectors", {})
        for label in ("manifest", "input", "rope_coefficients"):
            content_matches(
                Path(vectors[label]["path"]),
                vectors[label],
                f"layer {layer_index} vector {label}",
            )
        for tensor in vectors.get("tensors", []):
            content_matches(
                Path(tensor["serialized"]["path"]),
                tensor["serialized"],
                f"layer {layer_index} serialized tensor",
            )
        raw = layer.get("raw", {})
        for label in ("terminal", "trace"):
            content_matches(
                Path(raw[label]["path"]),
                raw[label],
                f"layer {layer_index} raw {label}",
            )
        counts = parse_natural_terminal(Path(raw["terminal"]["path"]), layer_index)
        require(
            all(raw.get(name) == value for name, value in counts.items()),
            f"layer {layer_index} terminal count binding mismatch",
        )
        output_sha256 = semantic_hidden_sha256(Path(layer["output"]["path"]))
        require(
            layer["output"].get("semantic_sha256") == output_sha256,
            f"layer {layer_index} semantic output binding mismatch",
        )
        validate_exact_transaction_artifacts(layer, layer_index, POSITION)
        previous_sha256 = output_sha256
    compared = compare_live_layers(layers, embedding_bits, write_reports=False)
    require(compared == layers, "stored live traversal comparisons are stale")
    require(
        traversal.get("post_layer23")
        == {
            "hidden_sha256": previous_sha256,
            "natural_terminal": True,
            "independent_oracle_within_tolerance": True,
        },
        "post-layer-23 terminal binding mismatch",
    )
    return dict(traversal)


def assemble_evidence(
    embedding: Mapping[str, Any],
    traversal: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "kind": "ace3_selected_token_position2_fresh_traversal_evidence",
        "status": "COMPLETE",
        "model": {
            "repository": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
            "revision": "db09cd27ead7fee40cdee309693cf83601b9c899",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "numeric_profile": (
                "native asymmetric packed INT4 AWQ W4A16 G128, no qzero "
                "plus-one adjustment, FP16 activations and FP16 K/V"
            ),
        },
        "selected_token": {
            "token_id": SELECTED_TOKEN_ID,
            "source": "authenticated original checkpoint embedding",
            "embedding_sha256": embedding["sha256"],
        },
        "official_tied_embedding": dict(embedding),
        "current_continuation_attempt": dict(traversal),
        "consumed_sources": consumed_source_records(),
        "claim_boundary": {
            "demonstrated": (
                "authenticated position-2 token 271 official FP16 embedding "
                "through a current-worktree 24-layer position-2 Verilator "
                "transaction chain with per-layer FP16 K/V lineage and "
                "independent-oracle comparisons"
            ),
            "lm_head": "not run",
            "dialogue_quality": "not claimed",
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
            "latency": "not measured",
            "throughput": "not measured",
        },
    }


def _write_exact_oracle(
    transaction_dir: Path,
    final: Sequence[int],
    trace: Sequence[tuple[int, int, int, int]],
) -> tuple[dict[str, Any], bytes, bytes]:
    oracle_dir = transaction_dir / "exact_oracle"
    oracle_dir.mkdir()
    trace_payload = oracle_trace_payload(trace)
    final_payload = hidden_payload(np.asarray(final, dtype="<u2"))
    trace_path = oracle_dir / "trace.hex"
    final_path = oracle_dir / "final.hex"
    trace_path.write_bytes(trace_payload)
    final_path.write_bytes(final_payload)
    return (
        {
            "implementation": "independent decoder_layer0_oracle.run_token",
            "trace": file_record(trace_path),
            "final_hidden": file_record(final_path),
        },
        trace_payload,
        final_payload,
    )


def execute_exact_transaction(
    binary: Path,
    layer_index: int,
    position: int,
    hidden_bits: np.ndarray,
    vectors: Mapping[str, Any],
    tensor_vector_dir: Path,
    transaction_dir: Path,
    state_out: Path,
    oracle_values: Mapping[str, Sequence[int]],
    cache_k: list[list[int]],
    cache_v: list[list[int]],
    state_in: Path | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    transaction, output_bits = execute_transaction(
        binary,
        layer_index,
        position,
        hidden_bits,
        vectors,
        tensor_vector_dir,
        transaction_dir,
        state_out,
        state_in,
    )
    final, trace = run_decoder_layer_token(
        oracle_values,
        np.asarray(hidden_bits, dtype="<u2").tolist(),
        position,
        cache_k,
        cache_v,
        accurate_silu=True,
    )
    oracle, expected_trace, expected_final = _write_exact_oracle(
        transaction_dir,
        final,
        trace,
    )
    trace_comparison = exact_hex_comparison(
        Path(transaction["raw"]["trace"]["path"]).read_bytes(),
        expected_trace,
        "trace",
    )
    final_comparison = exact_hex_comparison(
        Path(transaction["output"]["path"]).read_bytes(),
        expected_final,
        "final_hidden",
    )
    require(
        trace_comparison["exact_match"] and final_comparison["exact_match"],
        f"layer {layer_index} position {position} exact oracle comparison mismatch",
    )
    return (
        {
            **transaction,
            "exact_oracle": oracle,
            "exact_comparison": {
                "trace": trace_comparison,
                "final_hidden": final_comparison,
            },
        },
        output_bits,
    )


def validate_exact_transaction_artifacts(
    transaction: Mapping[str, Any],
    layer_index: int,
    position: int,
) -> None:
    oracle = transaction.get("exact_oracle", {})
    for label in ("trace", "final_hidden"):
        record = oracle.get(label, {})
        content_matches(
            Path(record["path"]),
            record,
            f"layer {layer_index} position {position} exact oracle {label}",
        )
    trace_comparison = exact_hex_comparison(
        Path(transaction["raw"]["trace"]["path"]).read_bytes(),
        Path(oracle["trace"]["path"]).read_bytes(),
        "trace",
    )
    final_comparison = exact_hex_comparison(
        Path(transaction["output"]["path"]).read_bytes(),
        Path(oracle["final_hidden"]["path"]).read_bytes(),
        "final_hidden",
    )
    require(
        transaction.get("exact_comparison")
        == {
            "trace": trace_comparison,
            "final_hidden": final_comparison,
        }
        and trace_comparison["exact_match"]
        and final_comparison["exact_match"],
        f"layer {layer_index} position {position} exact comparison mismatch",
    )


def generate(output: Path) -> dict[str, Any]:
    embedding = official_embedding_binding()
    embedding_bits = selected_embedding_bits()
    traversal = execute_live_traversal(
        output.parent / "traversal",
        embedding_bits,
    )
    document = assemble_evidence(embedding, traversal)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(document))
    return document


def validate(output: Path) -> dict[str, Any]:
    stored = load_json(output)
    require(
        stored.get("schema_version") == 3
        and stored.get("kind")
        == "ace3_selected_token_position2_fresh_traversal_evidence"
        and stored.get("status") == "COMPLETE",
        "fresh traversal evidence identity mismatch",
    )
    stored_sources = stored.get("consumed_sources")
    require(
        isinstance(stored_sources, dict),
        "stored consumed source closure is missing",
    )
    require_consumed_source_bindings(stored_sources)
    embedding_bits = selected_embedding_bits()
    traversal = stored.get("current_continuation_attempt")
    require(
        isinstance(traversal, dict),
        "stored current continuation attempt is missing",
    )
    validated_traversal = validate_live_traversal(traversal, embedding_bits)
    fresh = assemble_evidence(
        official_embedding_binding(),
        validated_traversal,
    )
    require(stored == fresh, "stored continuation evidence is stale")
    return stored


def print_summary(document: Mapping[str, Any], output: Path) -> None:
    traversal = document["current_continuation_attempt"]
    layers = traversal["layers"]
    max_error = max(
        layer["independent_oracle_comparison"]["max_abs_error"]
        for layer in layers
    )
    print(
        "MODEL24_SELECTED_TOKEN_POSITION2_VALIDATION_PASS "
        f"status={document['status']} "
        f"selected_token={document['selected_token']['token_id']} "
        f"embedding_sha256={document['official_tied_embedding']['sha256']} "
        f"layers={len(layers)} layer_order=0..23 "
        f"fp16_kv_layers={len(layers)} "
        f"independent_oracle_max_abs_error={max_error:.9f} "
        "current_traversal=current_worktree_verilator natural_terminals=24 "
        "synthesis=not_run ppa=not_measured fpga=not_run "
        "latency=not_measured throughput=not_measured "
        f"evidence={output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=("generate", "validate"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if args.operation == "generate":
            document = generate(output)
        else:
            document = validate(output)
    except (ContinuationError, OSError, ValueError, KeyError) as error:
        raise SystemExit(
            f"MODEL24_SELECTED_TOKEN_POSITION2_VALIDATION_FAIL {error}"
        ) from error
    print_summary(document, output)


if __name__ == "__main__":
    main()
