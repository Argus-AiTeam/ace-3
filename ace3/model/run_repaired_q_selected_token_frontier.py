#!/usr/bin/env python3
"""Run the accepted repaired-Q selected-token path to the layer-21 frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors import safe_open

import decoder_layer0_oracle
import model24_persistent_kv_runtime as persistent
import replay_layer00_q_projection_repair as repair
import validate_selected_token_position2_traversal as transaction
from diagnose_layer01_position2_stage00 import (
    authenticated_trace,
    file_record,
    trace_stage,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ATTEMPT = (
    ROOT / "build/model24_persistent_kv_second_token_attempt002/traversal"
)
REPAIR_ATTEMPT = (
    ROOT
    / "build/model24_persistent_kv_second_token_layer00_q_repair_attempt004"
)
REFERENCE = ROOT / "build/independent_fp16_trajectory_20260906_1133/recovery001"
DEFAULT_OUTPUT = (
    ROOT
    / "build/model24_persistent_kv_selected_token_repaired_q_attempt001"
)
ATTEMPT_PATTERN = re.compile(
    r"model24_persistent_kv_selected_token_repaired_q_attempt[0-9]{3}"
)
COMMAND_RECEIPT = "run.command.txt"
MAX_LAYER = 21
POSITION = 2
RECOVERABLE_FAILURE_FRAGMENT = "rtl_comparison_layer04_position000.json"
STAGE_COUNTS = {
    0: decoder_layer0_oracle.HIDDEN,
    1: decoder_layer0_oracle.HIDDEN,
    2: decoder_layer0_oracle.KV_HEADS * decoder_layer0_oracle.HEAD_DIM,
    3: decoder_layer0_oracle.KV_HEADS * decoder_layer0_oracle.HEAD_DIM,
    4: decoder_layer0_oracle.HIDDEN,
    5: decoder_layer0_oracle.KV_HEADS * decoder_layer0_oracle.HEAD_DIM,
    6: decoder_layer0_oracle.KV_HEADS * decoder_layer0_oracle.HEAD_DIM,
    7: decoder_layer0_oracle.KV_HEADS * decoder_layer0_oracle.HEAD_DIM,
    8: decoder_layer0_oracle.HEADS * (POSITION + 1),
    9: decoder_layer0_oracle.HEADS * (POSITION + 1),
    10: decoder_layer0_oracle.HIDDEN,
    11: decoder_layer0_oracle.HIDDEN,
    12: decoder_layer0_oracle.HIDDEN,
    13: decoder_layer0_oracle.HIDDEN,
    14: decoder_layer0_oracle.INTERMEDIATE,
    15: decoder_layer0_oracle.INTERMEDIATE,
    16: decoder_layer0_oracle.INTERMEDIATE,
    17: decoder_layer0_oracle.HIDDEN,
    18: decoder_layer0_oracle.HIDDEN,
}
PERIODIC_STAGE_INDICES = {
    8: POSITION + 1,
    9: POSITION + 1,
}


class FrontierError(RuntimeError):
    """Raised when a repaired-Q frontier attempt fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FrontierError(message)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", "")))
    actual = file_record(path)
    require(
        actual["bytes"] == record.get("bytes")
        and actual["sha256"] == record.get("sha256"),
        f"file binding mismatch: {path}",
    )
    return path


def read_hex_bits(record: Mapping[str, Any], count: int) -> np.ndarray:
    path = verify_file_record(record)
    rows = path.read_text(encoding="ascii").splitlines()
    require(
        len(rows) == count and all(len(row) == 4 for row in rows),
        f"hex geometry mismatch: {path}",
    )
    values = np.asarray([int(row, 16) for row in rows], dtype="<u2")
    semantic_sha256 = record.get("semantic_sha256")
    if semantic_sha256 is not None:
        require(
            hashlib.sha256(values.tobytes()).hexdigest() == semantic_sha256,
            f"semantic binding mismatch: {path}",
        )
    return values


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value))


def write_hex(path: Path, values: np.ndarray) -> dict[str, Any]:
    payload = "".join(f"{int(value):04x}\n" for value in values).encode("ascii")
    path.write_bytes(payload)
    return {
        **file_record(path),
        "records": int(values.size),
        "semantic_sha256": hashlib.sha256(
            np.asarray(values, dtype="<u2").tobytes()
        ).hexdigest(),
    }


def trace_stages(
    trace: Sequence[tuple[int, int, int, int]],
) -> dict[int, np.ndarray]:
    stages: dict[int, list[tuple[int, int]]] = {}
    for stage, index, value, position in trace:
        require(position == POSITION, "accepted trace position mismatch")
        stages.setdefault(stage, []).append((index, value))
    require(
        set(stages) == set(STAGE_COUNTS),
        "accepted trace stage coverage mismatch",
    )
    normalized = {}
    for stage, records in stages.items():
        count = STAGE_COUNTS[stage]
        index_period = PERIODIC_STAGE_INDICES.get(stage)
        if index_period is None:
            records.sort()
            expected_indices = list(range(count))
        else:
            expected_indices = list(range(index_period)) * (
                count // index_period
            )
        require(
            [index for index, _ in records] == expected_indices,
            f"accepted trace stage {stage} index mismatch",
        )
        normalized[stage] = np.asarray(
            [value for _, value in records],
            dtype="<u2",
        )
    return normalized


def producer_authenticated_trace(
    layer_index: int,
    position: int,
) -> tuple[bytes, dict[str, Any]]:
    position_dir = (
        SOURCE_ATTEMPT
        / f"layer{layer_index:02d}"
        / f"position{position:03d}"
    )
    raw_trace = position_dir / "raw/trace.hex"
    oracle_trace = position_dir / "exact_oracle/trace.hex"
    vector_trace = position_dir / "vectors/trace.hex"
    boundary_path = position_dir / "vectors/boundary_manifest.json"
    terminal_path = position_dir / "raw/terminal.txt"
    boundary = json.loads(boundary_path.read_text(encoding="ascii"))
    require(
        boundary.get("kind") == "ace3_decoder_token_runtime_vector_contract"
        and boundary.get("layer_index") == layer_index
        and boundary.get("position") == position,
        f"producer boundary identity mismatch: layer {layer_index} position {position}",
    )
    expected_trace = boundary.get("trace", {})
    vector_record = file_record(vector_trace)
    require(
        vector_record["bytes"] == expected_trace.get("bytes")
        and vector_record["sha256"] == expected_trace.get("sha256"),
        f"producer vector binding mismatch: layer {layer_index} position {position}",
    )
    payload = raw_trace.read_bytes()
    require(
        payload == vector_trace.read_bytes()
        and payload == oracle_trace.read_bytes()
        and len(payload.splitlines()) == boundary.get("trace_records"),
        f"producer exact-oracle trace mismatch: layer {layer_index} position {position}",
    )
    terminal_fields = dict(
        field.split("=", 1)
        for field in terminal_path.read_text(encoding="ascii").split()
    )
    require(
        terminal_fields
        == {
            "schema": "ace3_decoder_token_transaction_v1",
            "layer_index": str(layer_index),
            "position": str(position),
            "natural_terminal": "1",
            "exit_code": "0",
            "trace_count": str(23324 + 28 * position),
            "final_count": "896",
            "done_count": "1",
        },
        f"producer natural-terminal mismatch: layer {layer_index} position {position}",
    )
    return payload, {
        "kind": "original_producer_exact_oracle_trace_binding",
        "boundary_manifest": file_record(boundary_path),
        "raw_trace": file_record(raw_trace),
        "exact_oracle_trace": file_record(oracle_trace),
        "vector_trace": vector_record,
        "terminal": file_record(terminal_path),
        "natural_terminal": True,
    }


def prefix_cache(
    layer_index: int,
) -> tuple[list[list[int]], list[list[int]], list[dict[str, Any]]]:
    cache_k = []
    cache_v = []
    bindings = []
    for position in range(POSITION):
        comparison_path = REFERENCE / (
            f"rtl_comparison_layer{layer_index:02d}_position{position:03d}.json"
        )
        if comparison_path.is_file():
            trace_path = (
                SOURCE_ATTEMPT
                / f"layer{layer_index:02d}"
                / f"position{position:03d}/raw/trace.hex"
            )
            trace = authenticated_trace(comparison_path, trace_path)
            binding = {
                "kind": "independent_policy_comparison_trace_binding",
                "comparison": file_record(comparison_path),
                "raw_trace": file_record(trace_path),
            }
        else:
            trace, binding = producer_authenticated_trace(layer_index, position)
        cache_k.append(
            trace_stage(trace, position, 6, 128).astype("<u2").tolist()
        )
        cache_v.append(
            trace_stage(trace, position, 7, 128).astype("<u2").tolist()
        )
        bindings.append(binding)
    return cache_k, cache_v, bindings


def recover_completed_prefix(
    attempt_root: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    require(
        ATTEMPT_PATTERN.fullmatch(attempt_root.name) is not None,
        "recovery attempt directory identity mismatch",
    )
    progress_path = attempt_root / "progress.json"
    failure_path = attempt_root / "failure.json"
    command_path = attempt_root / COMMAND_RECEIPT
    progress = json.loads(progress_path.read_text(encoding="ascii"))
    failure = json.loads(failure_path.read_text(encoding="ascii"))
    last_completed = progress.get("last_completed_layer")
    require(
        progress.get("status") == "RUNNING"
        and isinstance(last_completed, int)
        and last_completed >= 1
        and failure.get("status") == "FAIL"
        and failure.get("attempt_id") == attempt_root.name
        and RECOVERABLE_FAILURE_FRAGMENT in failure.get("error", ""),
        "recovery source is not the preserved authentication-gap attempt",
    )
    layers = []
    layer_bindings = []
    for layer_index in range(last_completed + 1):
        result_path = attempt_root / f"layer{layer_index:02d}/layer_result.json"
        layer = json.loads(result_path.read_text(encoding="ascii"))
        exact = layer.get("fresh_rtl_simulation", {}).get("exact_comparison", {})
        raw = layer["fresh_rtl_simulation"]["raw"]
        terminal_counts = transaction.parse_natural_terminal(
            Path(raw["terminal"]["path"]),
            layer_index,
        )
        require(
            layer.get("layer_index") == layer_index
            and exact.get("trace", {}).get("exact_match")
            and exact.get("final_hidden", {}).get("exact_match")
            and all(raw.get(name) == value for name, value in terminal_counts.items())
            and not any(
                comparison.get("failure_count")
                for comparison in layer.get("independent_policy_comparisons", [])
            ),
            f"recovered layer {layer_index} is not a closed zero-mismatch prefix",
        )
        critical_records = [
            layer["fresh_rtl_simulation"]["raw"]["trace"],
            layer["fresh_rtl_simulation"]["raw"]["terminal"],
            layer["fresh_rtl_simulation"]["output"],
            layer["fresh_rtl_simulation"]["runtime_oracle"]["trace"],
            layer["fresh_rtl_simulation"]["runtime_oracle"]["final_hidden"],
            layer["controlled_q_injection"]["next_layer_input"],
        ]
        for record in critical_records:
            verify_file_record(record)
        layers.append(layer)
        layer_bindings.append(
            {
                "layer_index": layer_index,
                "layer_result": file_record(result_path),
                "critical_artifacts": critical_records,
            }
        )
    injected_hidden = read_hex_bits(
        layers[-1]["controlled_q_injection"]["next_layer_input"],
        decoder_layer0_oracle.HIDDEN,
    )
    return layers, injected_hidden, {
        "kind": "additive_recovery_from_authenticated_completed_prefix",
        "source_attempt": str(attempt_root),
        "last_completed_layer": last_completed,
        "command_receipt": file_record(command_path),
        "progress": file_record(progress_path),
        "failure": file_record(failure_path),
        "layers": layer_bindings,
    }


def write_oracle(
    output: Path,
    final: Sequence[int],
    trace: Sequence[tuple[int, int, int, int]],
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    output.mkdir(parents=True)
    stages = trace_stages(trace)
    stage_records = {}
    for stage, values in stages.items():
        stage_records[f"stage{stage:02d}"] = write_hex(
            output / f"stage{stage:02d}.hex",
            values,
        )
    final_record = write_hex(
        output / "final.hex",
        np.asarray(final, dtype="<u2"),
    )
    return {
        "implementation": (
            "decoder_layer0_oracle.run_token with accepted shared binary64 "
            "dequantized Q projection and FP16RNE Q output"
        ),
        "trace_stages": stage_records,
        "final_hidden": final_record,
    }, stages


def stage_reference(
    layer_index: int,
    stage: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    return repair.reference_array(REFERENCE, layer_index, stage)


def compare_stages(
    layer_index: int,
    stages: Mapping[int, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    comparisons = []
    first_failure = None
    for stage in range(19):
        expected, reference_record = stage_reference(layer_index, stage)
        compared = repair.comparison(stages[stage], expected)
        row = {
            "layer_index": layer_index,
            "position": POSITION,
            "stage": stage,
            "reference": reference_record,
            **compared,
        }
        comparisons.append(row)
        if first_failure is None and compared["failure_count"]:
            first_failure = row
    return comparisons, first_failure


def exact_runtime_oracle(
    values: Mapping[str, Sequence[int]],
    hidden: np.ndarray,
    cache_k: list[list[int]],
    cache_v: list[list[int]],
) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    return decoder_layer0_oracle.run_token(
        dict(values),
        hidden.tolist(),
        POSITION,
        [row.copy() for row in cache_k],
        [row.copy() for row in cache_v],
        accurate_silu=True,
        accepted_q_projection=False,
    )


def accepted_oracle(
    values: Mapping[str, Sequence[int]],
    hidden: np.ndarray,
    cache_k: list[list[int]],
    cache_v: list[list[int]],
) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    return decoder_layer0_oracle.run_token(
        dict(values),
        hidden.tolist(),
        POSITION,
        [row.copy() for row in cache_k],
        [row.copy() for row in cache_v],
        accurate_silu=True,
        accepted_q_projection=True,
    )


def run_layer(
    checkpoint: Any,
    attempt_root: Path,
    layer_index: int,
    injected_hidden: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any] | None]:
    source_layer = SOURCE_ATTEMPT / f"layer{layer_index:02d}"
    layer_root = attempt_root / f"layer{layer_index:02d}"
    vector_dir = layer_root / "vectors"
    vectors = transaction.materialize_transaction_vectors(
        checkpoint,
        layer_index,
        POSITION,
        injected_hidden,
        vector_dir,
    )
    values = transaction.layer_oracle_values(checkpoint, layer_index, vectors)
    cache_k, cache_v, prefix_bindings = prefix_cache(layer_index)

    runtime_final, runtime_trace = exact_runtime_oracle(
        values,
        injected_hidden,
        cache_k,
        cache_v,
    )
    runtime_oracle, expected_trace, expected_final = transaction._write_exact_oracle(
        layer_root,
        runtime_final,
        runtime_trace,
    )
    runtime_vectors = {
        **vectors,
        **transaction.materialize_runtime_vector_contract(
            layer_index,
            POSITION,
            runtime_final,
            runtime_trace,
            vector_dir,
        ),
    }
    source_binary = (
        source_layer.parent
        / f"compiled/layer{layer_index}/obj_dir"
        / "Vace3_decoder_layer0_token_engine"
    )
    source_state = source_layer / "position002.state"
    require(source_binary.is_file(), f"source layer {layer_index} binary missing")
    require(source_state.is_file(), f"source layer {layer_index} K/V state missing")
    actual, actual_bits = transaction.execute_transaction(
        source_binary,
        layer_index,
        POSITION,
        injected_hidden,
        runtime_vectors,
        vector_dir,
        layer_root,
        layer_root / "position003.state",
        source_state,
    )
    trace_comparison = transaction.exact_hex_comparison(
        Path(actual["raw"]["trace"]["path"]).read_bytes(),
        expected_trace,
        "trace",
    )
    final_comparison = transaction.exact_hex_comparison(
        Path(actual["output"]["path"]).read_bytes(),
        expected_final,
        "final_hidden",
    )
    require(
        trace_comparison["exact_match"] and final_comparison["exact_match"],
        f"layer {layer_index} fresh RTL differs from integer runtime oracle",
    )

    accepted_final, accepted_trace = accepted_oracle(
        values,
        injected_hidden,
        cache_k,
        cache_v,
    )
    accepted_record, accepted_stages = write_oracle(
        layer_root / "accepted_q_oracle",
        accepted_final,
        accepted_trace,
    )
    stage_comparisons, first_failure = compare_stages(
        layer_index,
        accepted_stages,
    )
    actual_stages = trace_stages(runtime_trace)
    q_intervention = repair.comparison(
        accepted_stages[1],
        actual_stages[1],
    )
    final_intervention = repair.comparison(
        np.asarray(accepted_final, dtype="<u2"),
        actual_bits,
    )
    require(
        not np.array_equal(accepted_stages[1], actual_stages[1]),
        f"layer {layer_index} repaired Q intervention was vacuous",
    )
    record = {
        "layer_index": layer_index,
        "position": POSITION,
        "input": {
            "source": (
                "official token embedding"
                if layer_index == 0
                else "prior accepted-Q controlled injection"
            ),
            "semantic_sha256": hashlib.sha256(
                injected_hidden.astype("<u2").tobytes()
            ).hexdigest(),
            "vectors": vectors["input"],
        },
        "persistent_kv_predecessor": {
            "source_state": file_record(source_state),
            "source_positions": [0, 1],
            "source_attempt": str(SOURCE_ATTEMPT),
            "authentication": prefix_bindings,
            "k_rows": [
                hashlib.sha256(np.asarray(row, dtype="<u2").tobytes()).hexdigest()
                for row in cache_k
            ],
            "v_rows": [
                hashlib.sha256(np.asarray(row, dtype="<u2").tobytes()).hexdigest()
                for row in cache_v
            ],
        },
        "fresh_rtl_simulation": {
            **actual,
            "binary": file_record(source_binary),
            "runtime_oracle": runtime_oracle,
            "exact_comparison": {
                "trace": trace_comparison,
                "final_hidden": final_comparison,
            },
        },
        "controlled_q_injection": {
            "accepted_oracle": accepted_record,
            "q_projection_vs_unrepaired_rtl": q_intervention,
            "final_hidden_vs_unrepaired_rtl": final_intervention,
            "next_layer_input": accepted_record["final_hidden"],
            "rtl_output_used_as_next_layer_input": False,
        },
        "independent_policy_comparisons": stage_comparisons,
    }
    write_json(layer_root / "layer_result.json", record)
    return record, np.asarray(accepted_final, dtype="<u2"), first_failure


def snapshot(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"seal.json", "failure.json"}:
            continue
        record = file_record(path)
        record["relative_path"] = path.relative_to(root).as_posix()
        records.append(record)
    return records


def generate(
    attempt_root: Path,
    resume_attempt: Path | None = None,
) -> dict[str, Any]:
    require(
        ATTEMPT_PATTERN.fullmatch(attempt_root.name) is not None,
        "attempt directory does not match the immutable repaired-Q namespace",
    )
    if attempt_root.exists():
        require(
            {path.name for path in attempt_root.iterdir()} == {COMMAND_RECEIPT},
            "refusing to replace a nonempty attempt directory",
        )
    else:
        attempt_root.mkdir(parents=True)
    command_path = attempt_root / COMMAND_RECEIPT
    require(command_path.is_file(), "pre-execution command receipt is missing")

    repair_result = json.loads(
        (REPAIR_ATTEMPT / "result.json").read_text(encoding="ascii")
    )
    require(
        repair_result.get("status") == "PASS"
        and repair_result.get("attempt_id") == REPAIR_ATTEMPT.name
        and repair_result["repaired_comparisons"]["layer01_stage08"][
            "failure_count"
        ]
        == 0,
        "accepted attempt004 Q repair binding mismatch",
    )
    started = time.monotonic()
    recovered_prefix = None
    if resume_attempt is None:
        layers = []
        injected_hidden = persistent.token_embedding(persistent.INJECTED_TOKEN_ID)
    else:
        layers, injected_hidden, recovered_prefix = recover_completed_prefix(
            resume_attempt.resolve()
        )
    first_failure = None
    with safe_open(persistent.CHECKPOINT, framework="np") as checkpoint:
        for layer_index in range(len(layers), MAX_LAYER + 1):
            layer, injected_hidden, layer_failure = run_layer(
                checkpoint,
                attempt_root,
                layer_index,
                injected_hidden,
            )
            layers.append(layer)
            progress = {
                "schema_version": 1,
                "status": "RUNNING",
                "last_completed_layer": layer_index,
                "elapsed_seconds": time.monotonic() - started,
            }
            write_json(attempt_root / "progress.json", progress)
            if layer_failure is not None:
                first_failure = layer_failure
                break

    reached_layer = layers[-1]["layer_index"]
    layer01_stage08 = next(
        row
        for row in layers[1]["independent_policy_comparisons"]
        if row["stage"] == 8
    )
    require(
        layer01_stage08["failure_count"] == 0,
        "layer01 stage08 repair regressed",
    )
    if first_failure is None:
        require(reached_layer == MAX_LAYER, "frontier ended before layer21")
        status = "LAYER21_PASS_FRONTIER"
    elif reached_layer == MAX_LAYER:
        status = "LAYER21_MISMATCH_FRONTIER"
    else:
        status = "EARLIEST_MATERIAL_MISMATCH_LOCALIZED"
    result = {
        "schema_version": 1,
        "kind": "ace3_repaired_q_selected_token_frontier",
        "attempt_id": attempt_root.name,
        "status": status,
        "model": {
            "repository": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
            "revision": decoder_layer0_oracle.REVISION,
            "numeric_profile": (
                "native asymmetric packed INT4 AWQ W4A16 G128, native GEMM "
                "nibble order, no qzero plus-one, FP16 activations and K/V"
            ),
        },
        "input_token_history": list(persistent.INPUT_TOKEN_HISTORY),
        "accepted_repair": file_record(REPAIR_ATTEMPT / "result.json"),
        "independent_reference": file_record(REFERENCE / "REPORT.json"),
        "command_receipt": file_record(command_path),
        "recovered_prefix": recovered_prefix,
        "reached_layer": reached_layer,
        "layer01_stage08": layer01_stage08,
        "earliest_material_mismatch": first_failure,
        "layers": layers,
        "elapsed_seconds": time.monotonic() - started,
        "evidence_level": "L2_SOFTWARE_CONTROLLED_INJECTION_DIAGNOSTIC",
        "accepted_repair_kind": "software numerical repair/replay",
        "actual_output_fed_rtl_chain": False,
        "claim_boundary": (
            "L2 computer-local selected-token position-2 diagnostic: each "
            "reported fresh Verilator transaction uses the old compiled RTL "
            "binary and is checked against its unrepaired integer runtime "
            "oracle, while software accepted-Q oracle output feeds the next "
            "layer. This is not an accepted repaired-RTL trajectory or a new "
            "dialogue token. No synthesis, PPA, bitstream, FPGA, or "
            "deployed-hardware claim."
        ),
    }
    write_json(attempt_root / "result.json", result)
    write_json(
        attempt_root / "status.json",
        {
            "schema_version": 1,
            "attempt_id": attempt_root.name,
            "status": status,
            "reached_layer": reached_layer,
            "layer01_stage08_failure_count": layer01_stage08["failure_count"],
            "earliest_material_mismatch": (
                None
                if first_failure is None
                else {
                    "layer_index": first_failure["layer_index"],
                    "stage": first_failure["stage"],
                    "failure_count": first_failure["failure_count"],
                    "first_failure": first_failure["first_failure"],
                }
            ),
        },
    )
    write_json(
        attempt_root / "seal.json",
        {
            "schema_version": 1,
            "kind": "ace3_repaired_q_selected_token_frontier_seal",
            "attempt_id": attempt_root.name,
            "status": "ENGINEERING_RESULT_REVIEW_REQUIRED",
            "command": {
                "argv": sys.argv,
                "cwd": str(Path.cwd()),
                "python": sys.version.split()[0],
                "host": platform.node(),
            },
            "artifacts": snapshot(attempt_root),
        },
    )
    return result


def validate(attempt_root: Path) -> dict[str, Any]:
    result = json.loads((attempt_root / "result.json").read_text(encoding="ascii"))
    require(
        result.get("kind") == "ace3_repaired_q_selected_token_frontier"
        and result.get("attempt_id") == attempt_root.name
        and result.get("evidence_level")
        == "L2_SOFTWARE_CONTROLLED_INJECTION_DIAGNOSTIC"
        and result.get("actual_output_fed_rtl_chain") is False
        and result.get("status")
        in {
            "LAYER21_PASS_FRONTIER",
            "LAYER21_MISMATCH_FRONTIER",
            "EARLIEST_MATERIAL_MISMATCH_LOCALIZED",
        },
        "result identity mismatch",
    )
    layers = result.get("layers")
    require(
        isinstance(layers, list)
        and [row["layer_index"] for row in layers]
        == list(range(result["reached_layer"] + 1)),
        "ordered layer frontier mismatch",
    )
    require(
        result["reached_layer"] >= 1
        and result["layer01_stage08"]["failure_count"] == 0,
        "layer01 stage08 is not materially closed",
    )
    for layer in layers:
        exact = layer["fresh_rtl_simulation"]["exact_comparison"]
        raw = layer["fresh_rtl_simulation"]["raw"]
        terminal_counts = transaction.parse_natural_terminal(
            Path(raw["terminal"]["path"]),
            layer["layer_index"],
        )
        require(
            exact["trace"]["exact_match"]
            and exact["final_hidden"]["exact_match"]
            and all(raw.get(name) == value for name, value in terminal_counts.items()),
            f"layer {layer['layer_index']} fresh RTL evidence mismatch",
        )
        for record in (
            layer["fresh_rtl_simulation"]["raw"]["trace"],
            layer["fresh_rtl_simulation"]["raw"]["terminal"],
            layer["fresh_rtl_simulation"]["output"],
            layer["fresh_rtl_simulation"]["runtime_oracle"]["trace"],
            layer["fresh_rtl_simulation"]["runtime_oracle"]["final_hidden"],
            layer["controlled_q_injection"]["next_layer_input"],
        ):
            verify_file_record(record)
    recovered_prefix = result.get("recovered_prefix")
    if recovered_prefix is not None:
        for key in ("command_receipt", "progress", "failure"):
            verify_file_record(recovered_prefix[key])
        for layer_binding in recovered_prefix["layers"]:
            verify_file_record(layer_binding["layer_result"])
            for record in layer_binding["critical_artifacts"]:
                verify_file_record(record)
    seal = json.loads((attempt_root / "seal.json").read_text(encoding="ascii"))
    require(
        seal.get("kind") == "ace3_repaired_q_selected_token_frontier_seal"
        and seal.get("attempt_id") == attempt_root.name,
        "seal identity mismatch",
    )
    actual = {
        record["relative_path"]: record
        for record in snapshot(attempt_root)
    }
    expected = {
        record["relative_path"]: record
        for record in seal.get("artifacts", [])
    }
    require(actual == expected, "sealed artifact closure changed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("generate", "validate"))
    parser.add_argument("--attempt-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-attempt", type=Path)
    args = parser.parse_args()
    attempt_root = args.attempt_dir.resolve()
    try:
        result = (
            generate(attempt_root, args.resume_attempt)
            if args.operation == "generate"
            else validate(attempt_root)
        )
    except Exception as error:
        if args.operation == "generate" and attempt_root.is_dir():
            failure = attempt_root / "failure.json"
            if not failure.exists():
                write_json(
                    failure,
                    {
                        "schema_version": 1,
                        "status": "FAIL",
                        "attempt_id": attempt_root.name,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
        print(
            f"REPAIRED_Q_SELECTED_TOKEN_FRONTIER_FAIL "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(
        "REPAIRED_Q_SELECTED_TOKEN_FRONTIER "
        f"status={result['status']} "
        f"attempt={result['attempt_id']} "
        f"reached_layer={result['reached_layer']} "
        f"layer01_stage08_failures="
        f"{result['layer01_stage08']['failure_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
