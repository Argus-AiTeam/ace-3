#!/usr/bin/env python3
"""Run token 0 through persistent Model24 RTL K/V state and the tied head."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from safetensors import safe_open

import model24_host_runtime as host_runtime
import run_final_rmsnorm_from_layer23 as final_rmsnorm
import run_tied_lm_head_topk_from_final_rmsnorm as tied_head
import validate_selected_token_position2_traversal as transaction_engine


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "build/model24_rtl_cascade/checkpoint/model.safetensors"
TOKENIZER_DIR = ROOT / "build/host_dialogue_audit_20260829T182819Z/tokenizer"
HOST_ATTEMPT = ROOT / "build/model24_tokenizer_host_integration_attempt001"
TIED_HEAD_PREDECESSOR = ROOT / "build/model24_tied_lm_head_topk_attempt002"
HOST_REVIEW = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
    "177b77ce57da/round-0001.json"
)
HOST_REVIEW_MISSION = HOST_REVIEW.parent / "mission.json"
DEFAULT_ATTEMPT = ROOT / "build/model24_persistent_kv_second_token_attempt002"
ATTEMPT_PATTERN = re.compile(r"model24_persistent_kv_second_token_attempt[0-9]{3}")
INDEPENDENT_POLICY_ROOT = (
    ROOT / "build/independent_fp16_trajectory_20260906_1133"
)
INDEPENDENT_POLICY_SUMMARY = INDEPENDENT_POLICY_ROOT / "SUMMARY.json"
INDEPENDENT_POLICY_REPORT = (
    INDEPENDENT_POLICY_ROOT / "recovery001/REPORT.json"
)

INPUT_TOKEN_HISTORY = (9707, 1879, 0)
PREFIX_TOKEN_IDS = INPUT_TOKEN_HISTORY[:2]
INJECTED_TOKEN_ID = INPUT_TOKEN_HISTORY[-1]
RTL_POSITION_INDEX = 2
POSITION_ORDINAL = 3
LAYER_COUNT = 24
HIDDEN_SIZE = 896
TOKEN_EMBEDDING_SHA256 = {
    9707: "150deee94cfa96d5e9342ca4e5041b2b662ab649e7174e4280a0a1b062d10d06",
    1879: "e53571236e61d5517340953dbd4ac3cb0cc2acb144ff42c8017d02f33bb308a8",
    0: "ec976f71606ab3e3754a4e66a71cb48b99361679215310fce9d29195b4bbd81e",
}
EVIDENCE_NAME = "evidence.json"
STATUS_NAME = "status.json"
TIMING_NAME = "timing.json"
SEAL_NAME = "sealed_manifest.json"

SOURCE_PATHS = (
    "ace3/model/model24_persistent_kv_runtime.py",
    "ace3/model/model24_host_runtime.py",
    "ace3/model/validate_selected_token_position2_traversal.py",
    "ace3/model/official_model24_dialogue.py",
    "ace3/model/run_final_rmsnorm_from_layer23.py",
    "ace3/model/run_tied_lm_head_topk_from_final_rmsnorm.py",
    "ace3/model/streaming_lm_head_reference.py",
    "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
    "ace3/rtl/ace3_fp16_kv_cache.sv",
    "ace3/rtl/ace3_fp16_rmsnorm_core.sv",
    "ace3/rtl/ace3_streaming_tied_lm_head_topk.sv",
    "ace3/tb/ace3_decoder_layer0_token_engine_main.cpp",
)

_ORIGINAL_COMPARE_LIVE_LAYERS = transaction_engine.compare_live_layers


class PersistentKvError(RuntimeError):
    """Raised when the bounded persistent-K/V attempt fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PersistentKvError(message)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicates,
    )
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    record: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if root is not None:
        record["relative_path"] = str(resolved.relative_to(root.resolve()))
    return record


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value))


def source_records() -> list[dict[str, Any]]:
    return [file_record(ROOT / relative) for relative in SOURCE_PATHS]


def authenticate_host_predecessor() -> dict[str, Any]:
    summary = host_runtime.validate_tied_head_tokenizer_host_integration_directory(
        HOST_ATTEMPT,
        TIED_HEAD_PREDECESSOR,
        TOKENIZER_DIR,
    )
    require(
        summary["resulting_token_history"] == list(INPUT_TOKEN_HISTORY)
        and summary["selected_token_id"] == INJECTED_TOKEN_ID,
        "reviewed tokenizer/host token history mismatch",
    )
    review = load_json(HOST_REVIEW)
    mission = load_json(HOST_REVIEW_MISSION)
    require(
        review.get("kind") == "round_reviewed_handoff"
        and review.get("producer_role") == "reviewer"
        and review.get("review", {}).get("status") == "done"
        and review.get("mission_context") == str(HOST_REVIEW_MISSION)
        and mission.get("node_key")
        == "w4a16-tokenizer-host-integration-from-lm-head-receipt"
        and "model24_tied_lm_head_topk_attempt002"
        in str(mission.get("objective")),
        "tokenizer/host independent review binding mismatch",
    )
    return {
        "attempt_id": host_runtime.TIED_HEAD_INTEGRATION_ATTEMPT_ID,
        "evidence": file_record(HOST_ATTEMPT / "integration.json"),
        "manifest": file_record(HOST_ATTEMPT / "manifest.json"),
        "independent_review": file_record(HOST_REVIEW),
        "review_mission": file_record(HOST_REVIEW_MISSION),
        "review_status": "DONE",
        "tied_head_attempt_id": host_runtime.TIED_HEAD_ATTEMPT_ID,
        "tied_head_seal": file_record(
            TIED_HEAD_PREDECESSOR / "sealed_output_manifest.json"
        ),
        "validated_summary": summary,
    }


def _configured_comparison(
    layers: Sequence[Mapping[str, Any]],
    embedding_bits: np.ndarray,
    *,
    write_reports: bool,
) -> list[dict[str, Any]]:
    compared = _ORIGINAL_COMPARE_LIVE_LAYERS(
        layers,
        embedding_bits,
        write_reports=write_reports,
    )
    for layer in compared:
        layer["independent_oracle_comparison"]["seed"] = (
            "authenticated token-0 tied embedding"
        )
    return compared


@contextmanager
def configured_transaction_engine() -> Iterator[None]:
    names = (
        "POSITION0_TOKEN_ID",
        "POSITION1_TOKEN_ID",
        "SELECTED_TOKEN_ID",
        "POSITION",
        "POSITION0_EMBEDDING_SHA256",
        "POSITION1_EMBEDDING_SHA256",
        "EMBEDDING_SHA256",
        "compare_live_layers",
    )
    saved = {name: getattr(transaction_engine, name) for name in names}
    transaction_engine.POSITION0_TOKEN_ID = PREFIX_TOKEN_IDS[0]
    transaction_engine.POSITION1_TOKEN_ID = PREFIX_TOKEN_IDS[1]
    transaction_engine.SELECTED_TOKEN_ID = INJECTED_TOKEN_ID
    transaction_engine.POSITION = RTL_POSITION_INDEX
    transaction_engine.POSITION0_EMBEDDING_SHA256 = TOKEN_EMBEDDING_SHA256[
        PREFIX_TOKEN_IDS[0]
    ]
    transaction_engine.POSITION1_EMBEDDING_SHA256 = TOKEN_EMBEDDING_SHA256[
        PREFIX_TOKEN_IDS[1]
    ]
    transaction_engine.EMBEDDING_SHA256 = TOKEN_EMBEDDING_SHA256[
        INJECTED_TOKEN_ID
    ]
    transaction_engine.compare_live_layers = _configured_comparison
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(transaction_engine, name, value)


def token_embedding(token_id: int) -> np.ndarray:
    with safe_open(CHECKPOINT, framework="np") as checkpoint:
        bits = np.asarray(
            checkpoint.get_tensor("model.embed_tokens.weight")[token_id],
            dtype="<f2",
        ).view("<u2")
    require(
        sha256_bytes(bits.tobytes()) == TOKEN_EMBEDDING_SHA256[token_id],
        f"official token-{token_id} embedding authentication failed",
    )
    return bits


def _trace_stage_sha256(path: Path, stage: int) -> str:
    rows = path.read_bytes().splitlines()
    selected: list[tuple[int, int]] = []
    for row in rows:
        require(
            len(row) == 16 and re.fullmatch(rb"[0-9a-f]{16}", row) is not None,
            f"malformed exact trace row: {path}",
        )
        position = int(row[2:6], 16)
        row_stage = int(row[6:8], 16)
        if row_stage == stage:
            require(
                position == RTL_POSITION_INDEX,
                "K/V update trace position mismatch",
            )
            selected.append((int(row[8:12], 16), int(row[12:16], 16)))
    require(
        len(selected) == 128
        and [index for index, _ in selected] == list(range(128)),
        "K/V update trace coverage mismatch",
    )
    payload = np.asarray([value for _, value in selected], dtype="<u2").tobytes()
    return sha256_bytes(payload)


def cache_lineage(traversal: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for layer_index, layer in enumerate(traversal["layers"]):
        trace = Path(layer["exact_oracle"]["trace"]["path"])
        parentage = layer["fp16_kv_parentage"]
        result.append(
            {
                "layer_index": layer_index,
                "source_position_count": 2,
                "target_position_count": POSITION_ORDINAL,
                "rtl_position_index": RTL_POSITION_INDEX,
                "predecessor_state": layer["position01_state"],
                "updated_state": layer["output_state"],
                "parent_k_sha256": parentage["k_tensor_sha256"],
                "parent_v_sha256": parentage["v_tensor_sha256"],
                "appended_k_row_sha256": _trace_stage_sha256(trace, 6),
                "appended_v_row_sha256": _trace_stage_sha256(trace, 7),
                "exact_integer_oracle_match": (
                    layer["exact_comparison"]["trace"]["exact_match"]
                    and layer["exact_comparison"]["final_hidden"]["exact_match"]
                ),
                "natural_terminal": True,
            }
        )
    require(
        [row["layer_index"] for row in result] == list(range(LAYER_COUNT)),
        "persistent K/V lineage is not complete and ordered",
    )
    return result


def _prepare_final_rmsnorm_output(attempt_root: Path) -> Path:
    output = attempt_root / "final_rmsnorm"
    for relative in ("vectors", "source", "tmp", "raw", "oracle"):
        (output / relative).mkdir(parents=True, exist_ok=False)
    return output


def run_final_rmsnorm(
    attempt_root: Path,
    traversal: Mapping[str, Any],
) -> dict[str, Any]:
    output = _prepare_final_rmsnorm_output(attempt_root)
    previous = transaction_engine.load_hidden_bits(
        Path(
            traversal["layers"][-1]["fresh_prefix"]["position1"]["output"][
                "path"
            ]
        )
    )
    current = transaction_engine.load_hidden_bits(
        Path(traversal["layers"][-1]["output"]["path"])
    )
    inputs = np.stack((previous, current))
    with safe_open(CHECKPOINT, framework="np") as checkpoint:
        weights = np.asarray(
            checkpoint.get_tensor("model.norm.weight"),
            dtype="<f2",
        ).view("<u2")
    final_rmsnorm.write_hex(output / "vectors/layer23_residual.hex", inputs)
    final_rmsnorm.write_hex(output / "vectors/model_norm_weight.hex", weights)
    harness = output / "source/final_rmsnorm_main.cpp"
    harness.write_text(final_rmsnorm.harness_source(), encoding="ascii")

    old_output = final_rmsnorm.OUTPUT
    final_rmsnorm.OUTPUT = output
    try:
        binary, compile_record = final_rmsnorm.compile_rtl(
            ROOT / "ace3/rtl/ace3_fp16_fixed.sv",
            ROOT / "ace3/rtl/ace3_fp16_rmsnorm_core.sv",
            harness,
        )
        simulation = final_rmsnorm.execute_rtl(binary)
        gate = final_rmsnorm.natural_terminal_gate(simulation)
        actual = final_rmsnorm.parse_layer23_residual(output / "raw/final.hex")
        integer = final_rmsnorm.integer_comparison(
            inputs,
            weights,
            actual,
            ROOT / "ace3/model/fp16_adaptation_oracle.py",
            gate,
        )
        policy = final_rmsnorm.fp16_policy_comparison(
            inputs,
            weights,
            actual,
            gate,
        )
    finally:
        final_rmsnorm.OUTPUT = old_output
    require(
        integer["bit_exact"]
        and integer["failure_count"] == 0
        and policy["within_tolerance"]
        and policy["failure_count"] == 0,
        "final RMSNorm comparison failed",
    )
    return {
        "status": "PASS",
        "consumed_token_index": 1,
        "input_hidden": file_record(output / "vectors/layer23_residual.hex"),
        "output": file_record(output / "raw/final.hex"),
        "terminal": gate,
        "compile": compile_record,
        "simulation": simulation,
        "integer_comparison": integer,
        "fp16_policy_comparison": policy,
    }


def run_tied_head(
    attempt_root: Path,
    final_result: Mapping[str, Any],
) -> dict[str, Any]:
    output = attempt_root / "tied_lm_head_topk"
    for relative in (
        "vectors",
        "source",
        "compiled",
        "tmp",
        "raw",
        "oracle",
        "negative",
        "negative/raw",
    ):
        (output / relative).mkdir(parents=True, exist_ok=False)
    normalized = final_rmsnorm.parse_layer23_residual(
        Path(final_result["output"]["path"])
    )[1]
    model = tied_head.authenticate_checkpoint()
    tied_head.write_lines(
        output / "vectors/hidden.hex",
        (f"{int(bits):04x}" for bits in normalized),
    )
    embedding = model["streamed_tensor"]
    tied_head.write_lines(
        output / "vectors/run.cfg",
        (
            f"checkpoint_bytes={tied_head.CHECKPOINT_BYTES}",
            f"weight_offset={embedding['absolute_offset']}",
            f"weight_bytes={embedding['bytes']}",
            f"hidden_size={HIDDEN_SIZE}",
            f"vocab_size={tied_head.VOCAB_SIZE}",
            f"top_k={tied_head.TOP_K}",
        ),
    )
    write_json(
        output / "input_manifest.json",
        {
            "schema_version": 1,
            "kind": "ace3_model24_persistent_kv_tied_head_inputs",
            "final_rmsnorm": final_result["output"],
            "consumed_token_index": 1,
            "official_model": model,
        },
    )

    old_output = tied_head.OUTPUT
    old_attempt = tied_head.ATTEMPT_ID
    tied_head.OUTPUT = output
    tied_head.ATTEMPT_ID = f"{attempt_root.name}_tied_lm_head"
    try:
        binary, compile_record, protocol = tied_head.compile_and_protocol()
        negative = tied_head.negative_gate_probes(binary)
        simulation = tied_head.run_official_simulation(binary)
        gate, bits, accumulators, top_k = tied_head.natural_terminal_gate(
            simulation
        )
        oracle, comparison = tied_head.oracle_comparison(
            normalized,
            model,
            gate,
            bits,
            accumulators,
            top_k,
        )
    finally:
        tied_head.OUTPUT = old_output
        tied_head.ATTEMPT_ID = old_attempt
    require(
        comparison["status"] == "PASS"
        and comparison["failure_count"] == 0
        and comparison["selected_token_logit_match"] is True,
        "tied lm_head/Top-K comparison failed",
    )
    return {
        "status": "PASS",
        "input": file_record(output / "vectors/hidden.hex"),
        "compile": compile_record,
        "protocol_simulation": protocol,
        "negative_terminal_gates": negative,
        "simulation": simulation,
        "terminal": gate,
        "oracle": oracle,
        "comparison": comparison,
        "top_k": oracle["top_k_entries"],
        "selected_token_id": comparison["selected_token_id"],
        "selected_logit_f16_bits": comparison["selected_logit_f16_bits"],
    }


def host_continuation(tied_result: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer = host_runtime.authenticate_tokenizer(TOKENIZER_DIR)
    selected = tied_result["selected_token_id"]
    history = [*INPUT_TOKEN_HISTORY, selected]
    return {
        "input_token_history": list(INPUT_TOKEN_HISTORY),
        "selected_token_id": selected,
        "selected_logit_f16_bits": tied_result["selected_logit_f16_bits"],
        "resulting_token_history": history,
        "decoded_selected_token": tokenizer.decode(
            [selected], skip_special_tokens=False
        ),
        "decoded_transcript": tokenizer.decode(
            history, skip_special_tokens=False
        ),
    }


def _recover_vectors(vector_dir: Path, *, full: bool) -> dict[str, Any]:
    result: dict[str, Any]
    if full:
        manifest_path = vector_dir / "manifest.json"
        manifest = load_json(manifest_path)
        require(
            manifest.get("schema_version") == 1
            and manifest.get("kind") == "ace3_position2_live_transaction_vectors",
            f"decoder vector manifest identity mismatch: {manifest_path}",
        )
        result = {
            "manifest": file_record(manifest_path),
            "input": manifest["input"],
            "rope_coefficients": manifest["rope_coefficients"],
            "tensors": manifest["tensors"],
        }
    else:
        result = {
            "input": file_record(vector_dir / "inputs.hex"),
            "rope_coefficients": file_record(
                vector_dir / "rope_coefficients.hex"
            ),
        }
    boundary_path = vector_dir / "boundary_manifest.json"
    boundary = load_json(boundary_path)
    require(
        boundary.get("schema_version") == 1
        and boundary.get("kind")
        == "ace3_decoder_token_runtime_vector_contract"
        and boundary.get("trace") == file_record(vector_dir / "trace.hex")
        and boundary.get("final_hidden")
        == file_record(vector_dir / "final.hex"),
        f"decoder runtime-vector boundary mismatch: {boundary_path}",
    )
    return {
        **result,
        "trace": boundary["trace"],
        "final_hidden": boundary["final_hidden"],
        "boundary_manifest": file_record(boundary_path),
    }


def _recover_transaction(
    transaction_dir: Path,
    layer_index: int,
    position: int,
    vectors: Mapping[str, Any],
    state_out: Path,
) -> dict[str, Any]:
    raw_dir = transaction_dir / "raw"
    output_path = raw_dir / "final.hex"
    terminal_path = raw_dir / "terminal.txt"
    trace_path = raw_dir / "trace.hex"
    counts = transaction_engine.parse_natural_terminal(
        terminal_path,
        layer_index,
        position,
    )
    exact_trace = transaction_dir / "exact_oracle/trace.hex"
    exact_final = transaction_dir / "exact_oracle/final.hex"
    trace_comparison = transaction_engine.exact_hex_comparison(
        trace_path.read_bytes(),
        exact_trace.read_bytes(),
        "trace",
    )
    final_comparison = transaction_engine.exact_hex_comparison(
        output_path.read_bytes(),
        exact_final.read_bytes(),
        "final_hidden",
    )
    require(
        trace_comparison["exact_match"] and final_comparison["exact_match"],
        f"layer {layer_index} position {position} recovered exact comparison failed",
    )
    return {
        "layer_index": layer_index,
        "position": position,
        "input": {
            "dtype": "FP16",
            "elements": HIDDEN_SIZE,
            "sha256": transaction_engine.semantic_hidden_sha256(
                Path(vectors["input"]["path"])
            ),
        },
        "output": {
            **file_record(output_path),
            "dtype": "FP16",
            "elements": HIDDEN_SIZE,
            "semantic_sha256": transaction_engine.semantic_hidden_sha256(
                output_path
            ),
        },
        "output_state": file_record(state_out),
        "vectors": dict(vectors),
        "simulation_log": file_record(transaction_dir / "simulation.log"),
        "raw": {
            "terminal": file_record(terminal_path),
            "trace": file_record(trace_path),
            **counts,
        },
        "exact_oracle": {
            "implementation": "independent decoder_layer0_oracle.run_token",
            "trace": file_record(exact_trace),
            "final_hidden": file_record(exact_final),
        },
        "exact_comparison": {
            "trace": trace_comparison,
            "final_hidden": final_comparison,
        },
    }


def recover_completed_traversal(source_attempt: Path) -> dict[str, Any]:
    source_attempt = source_attempt.resolve(strict=True)
    require(
        ATTEMPT_PATTERN.fullmatch(source_attempt.name) is not None,
        "decoder recovery source is not a persistent-K/V attempt",
    )
    failure = load_json(source_attempt / "failure.json")
    require(
        failure.get("status") == "FAIL"
        and failure.get("error_type") == "FileExistsError"
        and str(failure.get("error", "")).endswith(
            f"{source_attempt}/final_rmsnorm/compiled'"
        ),
        "decoder recovery source did not fail at the known post-decoder boundary",
    )
    live_root = source_attempt / "traversal"
    layers = []
    for layer_index in range(LAYER_COUNT):
        layer_dir = live_root / f"layer{layer_index:02d}"
        position0 = _recover_transaction(
            layer_dir / "position000",
            layer_index,
            0,
            _recover_vectors(layer_dir / "position000/vectors", full=False),
            layer_dir / "position001.state",
        )
        position1 = _recover_transaction(
            layer_dir / "position001",
            layer_index,
            1,
            _recover_vectors(layer_dir / "position001/vectors", full=False),
            layer_dir / "position002.state",
        )
        current = _recover_transaction(
            layer_dir,
            layer_index,
            RTL_POSITION_INDEX,
            _recover_vectors(layer_dir / "vectors", full=True),
            layer_dir / "position003.state",
        )
        layers.append(
            {
                "layer_index": layer_index,
                "position": RTL_POSITION_INDEX,
                "input": current["input"],
                "output": current["output"],
                "position01_state": position1["output_state"],
                "fresh_prefix": {
                    "position0": position0,
                    "position1": position1,
                },
                "output_state": current["output_state"],
                "vectors": current["vectors"],
                "live_binary": file_record(
                    live_root
                    / f"compiled/layer{layer_index}/obj_dir"
                    / "Vace3_decoder_layer0_token_engine"
                ),
                "compile_log": file_record(layer_dir / "compile.log"),
                "simulation_log": current["simulation_log"],
                "raw": current["raw"],
                "exact_oracle": current["exact_oracle"],
                "exact_comparison": current["exact_comparison"],
            }
        )
    with configured_transaction_engine():
        compared = transaction_engine.compare_live_layers(
            layers,
            token_embedding(INJECTED_TOKEN_ID),
            write_reports=False,
        )
        traversal = {
            "status": "COMPLETE",
            "execution": "current-worktree compiled Verilator RTL",
            "selected_token_id": INJECTED_TOKEN_ID,
            "position": RTL_POSITION_INDEX,
            "layer_order": list(range(LAYER_COUNT)),
            "natural_terminal_layers": LAYER_COUNT,
            "fresh_token_inputs": {
                "position0": {
                    "token_id": PREFIX_TOKEN_IDS[0],
                    "embedding_sha256": TOKEN_EMBEDDING_SHA256[
                        PREFIX_TOKEN_IDS[0]
                    ],
                },
                "position1": {
                    "token_id": PREFIX_TOKEN_IDS[1],
                    "embedding_sha256": TOKEN_EMBEDDING_SHA256[
                        PREFIX_TOKEN_IDS[1]
                    ],
                },
            },
            "layers": compared,
            "post_layer23": {
                "hidden_sha256": compared[-1]["output"]["semantic_sha256"],
                "natural_terminal": True,
                "independent_oracle_within_tolerance": True,
            },
        }
        return transaction_engine.validate_live_traversal(
            traversal,
            token_embedding(INJECTED_TOKEN_ID),
        )


def independent_policy_assessment(
    final_result: Mapping[str, Any],
    tied_result: Mapping[str, Any],
) -> dict[str, Any]:
    summary = load_json(INDEPENDENT_POLICY_SUMMARY)
    report = load_json(INDEPENDENT_POLICY_REPORT)
    require(
        summary.get("status") == "COMPLETE_INDEPENDENT_POLICY_TRAJECTORY"
        and report.get("status") == "COMPLETE_INDEPENDENT_POLICY_TRAJECTORY"
        and report.get("reference_seeds_from_rtl") is False
        and report.get("overall_rtl_within_tolerance") is False,
        "independent FP16 policy trajectory identity mismatch",
    )
    generation = report["generated"][1]
    require(
        generation["input_history"] == list(INPUT_TOKEN_HISTORY)
        and report.get("first_token_matches_observed_zero") is True,
        "independent FP16 policy token history mismatch",
    )
    expected_norm_record = report["generation1_final_norm"]
    expected_norm_path = Path(expected_norm_record["path"])
    require(
        sha256_file(expected_norm_path) == expected_norm_record["file_sha256"],
        "independent final RMSNorm array binding mismatch",
    )
    expected_norm = np.load(expected_norm_path, allow_pickle=False)
    require(
        expected_norm.dtype == np.dtype("<u2")
        and expected_norm.shape == (1, HIDDEN_SIZE),
        "independent final RMSNorm array geometry mismatch",
    )
    actual_norm = final_rmsnorm.parse_layer23_residual(
        Path(final_result["output"]["path"])
    )[1]
    terminal_comparison = compare_fp16_policy(
        actual_norm,
        expected_norm[0],
    )
    observed_top_k = tied_result["top_k"]
    expected_top_k = generation["top10"]
    top_k_ids_match = [
        row["token_id"] for row in observed_top_k
    ] == [row["token_id"] for row in expected_top_k]
    top_k_logits_match = [
        row["logit_f16_bits"] for row in observed_top_k
    ] == [row["logit_bits"] for row in expected_top_k]
    return {
        "status": "NUMERIC_MISMATCH",
        "policy": (
            "independently propagated accepted mathematical FP16-stage policy; "
            "not stock-framework equivalence and not the local component oracle"
        ),
        "reference_seeds_from_rtl": False,
        "summary": file_record(INDEPENDENT_POLICY_SUMMARY),
        "report": file_record(INDEPENDENT_POLICY_REPORT),
        "coverage": summary["coverage"],
        "first_rtl_mismatch": report["first_rtl_mismatch"],
        "terminal_final_rmsnorm": terminal_comparison,
        "tied_head": {
            "selected_token_id": tied_result["selected_token_id"],
            "reference_selected_token_id": generation["selected_token_id"],
            "selected_token_match": (
                tied_result["selected_token_id"]
                == generation["selected_token_id"]
            ),
            "selected_logit_f16_bits": tied_result[
                "selected_logit_f16_bits"
            ],
            "reference_selected_logit_f16_bits": generation["top10"][0][
                "logit_bits"
            ],
            "selected_logit_match": (
                tied_result["selected_logit_f16_bits"]
                == generation["top10"][0]["logit_bits"]
            ),
            "top_k_token_order_match": top_k_ids_match,
            "top_k_logit_bits_match": top_k_logits_match,
        },
        "overall_within_tolerance": False,
        "threshold_changed": False,
    }


def compare_fp16_policy(
    actual_bits: np.ndarray,
    expected_bits: np.ndarray,
) -> dict[str, Any]:
    require(
        actual_bits.dtype == np.dtype("<u2")
        and expected_bits.dtype == np.dtype("<u2")
        and actual_bits.shape == expected_bits.shape,
        "independent final RMSNorm comparison geometry mismatch",
    )
    actual_values = actual_bits.view("<f2").astype(np.float64)
    expected_values = expected_bits.view("<f2").astype(np.float64)
    require(
        np.isfinite(actual_values).all()
        and np.isfinite(expected_values).all(),
        "nonfinite independent final RMSNorm comparison operand",
    )
    absolute = np.abs(actual_values - expected_values)
    relative = absolute / np.maximum(np.abs(expected_values), 2.0**-24)
    ulp = np.asarray(
        [
            abs(
                final_rmsnorm.ordered_f16(int(actual))
                - final_rmsnorm.ordered_f16(int(expected))
            )
            for actual, expected in zip(
                actual_bits.flat,
                expected_bits.flat,
            )
        ],
        dtype=np.int64,
    ).reshape(actual_bits.shape)
    failed = np.flatnonzero(
        (absolute > final_rmsnorm.ABSOLUTE_TOLERANCE)
        & (relative > final_rmsnorm.RELATIVE_TOLERANCE)
        & (ulp > final_rmsnorm.MAX_ULP_DISTANCE)
    )
    first = None
    if failed.size:
        index = int(failed[0])
        first = {
            "element_index": index,
            "actual_f16_bits": f"{int(actual_bits.flat[index]):04x}",
            "expected_f16_bits": f"{int(expected_bits.flat[index]):04x}",
            "actual": float(actual_values.flat[index]),
            "expected": float(expected_values.flat[index]),
            "absolute_error": float(absolute.flat[index]),
            "relative_error": float(relative.flat[index]),
            "ulp_distance": int(ulp.flat[index]),
        }
    return {
        "records": int(actual_bits.size),
        "absolute_tolerance": final_rmsnorm.ABSOLUTE_TOLERANCE,
        "relative_tolerance": final_rmsnorm.RELATIVE_TOLERANCE,
        "max_ulp_distance_allowed": final_rmsnorm.MAX_ULP_DISTANCE,
        "material_failure_rule": (
            "absolute, relative, and ULP thresholds must all be exceeded"
        ),
        "failure_count": int(failed.size),
        "max_absolute_error": float(absolute.max()),
        "max_relative_error": float(relative.max()),
        "max_ulp_distance": int(ulp.max()),
        "within_tolerance": failed.size == 0,
        "first_material_mismatch": first,
    }


def _validate_core_contract(document: Mapping[str, Any]) -> None:
    require(
        document.get("schema_version") == 1
        and document.get("kind") == "ace3_model24_persistent_kv_second_token"
        and document.get("status") == "NUMERIC_MISMATCH_REVIEW_REQUIRED",
        "persistent-K/V evidence identity mismatch",
    )
    require(
        document.get("input", {}).get("token_history")
        == list(INPUT_TOKEN_HISTORY)
        and document.get("input", {}).get("injected_token_id")
        == INJECTED_TOKEN_ID
        and document.get("input", {}).get("position_ordinal")
        == POSITION_ORDINAL,
        "persistent-K/V token history or position drift",
    )
    lineage = document.get("cache_lineage")
    require(
        isinstance(lineage, list)
        and len(lineage) == LAYER_COUNT
        and [row.get("layer_index") for row in lineage]
        == list(range(LAYER_COUNT)),
        "persistent-K/V layer lineage mismatch",
    )
    for layer_index, row in enumerate(lineage):
        require(
            row.get("source_position_count") == 2
            and row.get("target_position_count") == POSITION_ORDINAL
            and row.get("rtl_position_index") == RTL_POSITION_INDEX
            and row.get("natural_terminal") is True
            and row.get("exact_integer_oracle_match") is True,
            f"layer {layer_index} persistent-K/V update mismatch",
        )
    tied = document.get("tied_lm_head_topk", {})
    continuation = document.get("host_continuation", {})
    require(
        tied.get("status") == "PASS"
        and tied.get("comparison", {}).get("failure_count") == 0
        and tied.get("comparison", {}).get("top_k_mismatch_count") == 0
        and tied.get("selected_token_id")
        == continuation.get("selected_token_id")
        and len(tied.get("top_k", [])) == tied_head.TOP_K,
        "terminal selected-token/logit/Top-K agreement drift",
    )
    policy = document.get("independent_policy_assessment", {})
    require(
        policy.get("status") == "NUMERIC_MISMATCH"
        and policy.get("reference_seeds_from_rtl") is False
        and policy.get("overall_within_tolerance") is False
        and policy.get("first_rtl_mismatch", {}).get("failure_count", 0) > 0,
        "independently propagated FP16 policy mismatch was not preserved",
    )


def focused_drift_checks(document: Mapping[str, Any]) -> dict[str, Any]:
    mutations = (
        (
            "token_history",
            lambda item: item["input"]["token_history"].append(1),
        ),
        (
            "cache_position",
            lambda item: item["cache_lineage"][0].__setitem__(
                "target_position_count", 2
            ),
        ),
        (
            "selected_token",
            lambda item: item["host_continuation"].__setitem__(
                "selected_token_id", -1
            ),
        ),
    )
    results = []
    for name, mutate in mutations:
        candidate = copy.deepcopy(document)
        mutate(candidate)
        try:
            _validate_core_contract(candidate)
        except PersistentKvError as error:
            results.append({"mutation": name, "rejected": True, "reason": str(error)})
        else:
            raise PersistentKvError(f"focused drift mutation was accepted: {name}")
    return {
        "status": "PASS",
        "checks": results,
        "rejected_count": len(results),
    }


def snapshot(root: Path) -> list[dict[str, Any]]:
    return [
        file_record(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != SEAL_NAME
    ]


def generate(
    attempt_root: Path,
    completed_decoder_attempt: Path | None = None,
) -> dict[str, Any]:
    require(
        ATTEMPT_PATTERN.fullmatch(attempt_root.name) is not None,
        "attempt directory name is not a fresh persistent-K/V attempt ID",
    )
    require(not attempt_root.exists(), f"attempt already exists: {attempt_root}")
    attempt_root.mkdir(parents=True)
    phase_seconds: dict[str, float] = {}
    started = time.monotonic()

    phase = time.monotonic()
    predecessor = authenticate_host_predecessor()
    embedding = token_embedding(INJECTED_TOKEN_ID)
    phase_seconds["authenticate_inputs"] = time.monotonic() - phase

    recovery = None
    phase = time.monotonic()
    if completed_decoder_attempt is None:
        with configured_transaction_engine():
            traversal = transaction_engine.execute_live_traversal(
                attempt_root / "traversal",
                embedding,
            )
        phase_seconds["decoder_rtl"] = time.monotonic() - phase
    else:
        completed_decoder_attempt = completed_decoder_attempt.resolve(
            strict=True
        )
        traversal = recover_completed_traversal(completed_decoder_attempt)
        phase_seconds["decoder_artifact_revalidation"] = (
            time.monotonic() - phase
        )
        receipt_path = (
            ROOT
            / ".argus_subagents"
            / f"ace3-model24-persistent-kv-second-token-"
            f"{completed_decoder_attempt.name.removeprefix(
                'model24_persistent_kv_second_token_'
            )}.json"
        )
        receipt = load_json(receipt_path)
        require(
            receipt.get("state") == "error"
            and receipt.get("exit_code") == 1,
            "completed decoder durable receipt identity mismatch",
        )
        recovery = {
            "status": "RECOVERED_COMPLETED_DECODER_ARTIFACTS",
            "source_attempt_id": completed_decoder_attempt.name,
            "source_failure": file_record(
                completed_decoder_attempt / "failure.json"
            ),
            "durable_receipt": file_record(receipt_path),
            "durable_run_id": receipt["run_id"],
            "durable_elapsed_seconds": receipt["elapsed_seconds"],
            "decoder_transactions_reexecuted": 0,
            "source_artifacts_mutated": False,
        }
    lineage = cache_lineage(traversal)

    phase = time.monotonic()
    final_result = run_final_rmsnorm(attempt_root, traversal)
    phase_seconds["final_rmsnorm_rtl"] = time.monotonic() - phase

    phase = time.monotonic()
    tied_result = run_tied_head(attempt_root, final_result)
    phase_seconds["tied_lm_head_topk_rtl_and_oracle"] = time.monotonic() - phase
    continuation = host_continuation(tied_result)
    policy_assessment = independent_policy_assessment(
        final_result,
        tied_result,
    )

    document = {
        "schema_version": 1,
        "kind": "ace3_model24_persistent_kv_second_token",
        "status": "NUMERIC_MISMATCH_REVIEW_REQUIRED",
        "attempt_id": attempt_root.name,
        "model": {
            "repository": tied_head.MODEL_REPOSITORY,
            "revision": tied_head.MODEL_REVISION,
            "checkpoint": file_record(CHECKPOINT),
            "numeric_profile": (
                "native asymmetric packed INT4 AWQ W4A16 G128, native GEMM "
                "nibble ordering, no qzero plus-one adjustment, FP16 "
                "activations, FP16 K/V"
            ),
        },
        "reviewed_predecessor": predecessor,
        "input": {
            "token_history": list(INPUT_TOKEN_HISTORY),
            "prefix_token_ids": list(PREFIX_TOKEN_IDS),
            "injected_token_id": INJECTED_TOKEN_ID,
            "rtl_position_index": RTL_POSITION_INDEX,
            "position_ordinal": POSITION_ORDINAL,
            "embedding_sha256": TOKEN_EMBEDDING_SHA256[INJECTED_TOKEN_ID],
        },
        "decoder_traversal": traversal,
        "cache_lineage": lineage,
        "terminal_hidden": {
            "layer23_output": traversal["layers"][-1]["output"],
            "exact_integer_oracle_match": traversal["layers"][-1][
                "exact_comparison"
            ]["final_hidden"]["exact_match"],
            "fp16_policy": traversal["layers"][-1][
                "independent_oracle_comparison"
            ],
        },
        "final_rmsnorm": final_result,
        "tied_lm_head_topk": tied_result,
        "host_continuation": continuation,
        "independent_policy_assessment": policy_assessment,
        "source_bindings": source_records(),
        "claim_boundary": {
            "demonstrated": (
                "computer-local position-3 persistent-K/V RTL simulation "
                "through 24 decoder layers, final RMSNorm, tied lm_head/Top-K, "
                "local exact-oracle comparisons, and tokenizer decoding"
            ),
            "not_demonstrated": (
                "complete independently propagated accepted-stage-policy "
                "numerical agreement"
            ),
            "dialogue_scope": "one additional deterministic greedy token",
            "synthesis": "not run",
            "ppa": "not measured",
            "bitstream": "not built",
            "fpga": "not run",
            "silicon": "not run",
        },
    }
    if recovery is not None:
        document["decoder_recovery"] = recovery
    _validate_core_contract(document)
    document["focused_drift_validation"] = focused_drift_checks(document)
    write_json(attempt_root / EVIDENCE_NAME, document)

    phase_seconds["total"] = time.monotonic() - started
    timing = {
        "schema_version": 1,
        "clock": "time.monotonic",
        "phases_seconds": phase_seconds,
        "performance_claim": (
            "measurement only; no bottleneck attribution or hardware claim"
        ),
    }
    write_json(attempt_root / TIMING_NAME, timing)
    status = {
        "schema_version": 1,
        "status": "NUMERIC_MISMATCH_REVIEW_REQUIRED",
        "attempt_id": attempt_root.name,
        "natural_terminal_layers": traversal["natural_terminal_layers"],
        "final_rmsnorm_natural_terminal": final_result["terminal"][
            "natural_terminal"
        ],
        "tied_head_natural_terminal": tied_result["terminal"][
            "natural_terminal"
        ],
        "local_exact_oracle_mismatches": 0,
        "independent_policy_first_mismatch": policy_assessment[
            "first_rtl_mismatch"
        ],
        "independent_final_rmsnorm_failure_count": policy_assessment[
            "terminal_final_rmsnorm"
        ]["failure_count"],
        "selected_token_id": tied_result["selected_token_id"],
        "selected_logit_f16_bits": tied_result["selected_logit_f16_bits"],
        "top_k_entries": len(tied_result["top_k"]),
        "decoded_transcript": continuation["decoded_transcript"],
    }
    write_json(attempt_root / STATUS_NAME, status)
    seal = {
        "schema_version": 1,
        "kind": "ace3_model24_persistent_kv_second_token_seal",
        "status": "ENGINEERING_RESULT_REVIEW_REQUIRED",
        "attempt_id": attempt_root.name,
        "command": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "python": sys.version.split()[0],
            "host": platform.node(),
        },
        "artifacts": snapshot(attempt_root),
    }
    write_json(attempt_root / SEAL_NAME, seal)
    return document


def _validate_file_record(record: Mapping[str, Any], label: str) -> None:
    path = Path(str(record.get("path", "")))
    require(
        path.is_file()
        and path.stat().st_size == record.get("bytes")
        and sha256_file(path) == record.get("sha256"),
        f"{label} file binding mismatch",
    )


def validate_final_rmsnorm(document: Mapping[str, Any]) -> None:
    result = document["final_rmsnorm"]
    output = Path(result["output"]["path"])
    _validate_file_record(result["output"], "final RMSNorm output")
    actual = final_rmsnorm.parse_layer23_residual(output)
    inputs = np.asarray(
        [
            int(row, 16)
            for row in Path(result["input_hidden"]["path"])
            .read_text(encoding="ascii")
            .splitlines()
        ],
        dtype="<u2",
    ).reshape(2, HIDDEN_SIZE)
    with safe_open(CHECKPOINT, framework="np") as checkpoint:
        weights = np.asarray(
            checkpoint.get_tensor("model.norm.weight"), dtype="<f2"
        ).view("<u2")
    oracle = final_rmsnorm.load_integer_oracle(
        ROOT / "ace3/model/fp16_adaptation_oracle.py"
    )
    expected = np.asarray(
        [
            [value for value, invalid, saturated in oracle.rmsnorm(
                row.tolist(), weights.tolist()
            )[0]]
            for row in inputs
        ],
        dtype="<u2",
    )
    require(
        np.array_equal(actual, expected),
        "fresh final RMSNorm integer semantic comparison failed",
    )


def validate_tied_head(document: Mapping[str, Any]) -> None:
    result = document["tied_lm_head_topk"]
    output = Path(result["input"]["path"]).parents[1]
    terminal = output / "raw/terminal.txt"
    match = tied_head.NATURAL_TERMINAL_RE.fullmatch(terminal.read_bytes())
    require(match is not None, "tied-head natural terminal changed")
    bits, accumulators = tied_head.parse_raw_logits(output / "raw/logits.txt")
    top_k = tied_head.parse_raw_topk(output / "raw/topk.txt")
    hidden = np.asarray(
        [
            int(row, 16)
            for row in Path(result["input"]["path"])
            .read_text(encoding="ascii")
            .splitlines()
        ],
        dtype="<u2",
    )
    model = tied_head.authenticate_checkpoint()
    with tempfile.TemporaryDirectory(prefix="ace3-persistent-kv-validate-") as temp:
        temporary = Path(temp)
        (temporary / "oracle").mkdir()
        (temporary / "raw").mkdir()
        for name in ("logits.txt", "topk.txt"):
            shutil.copyfile(output / "raw" / name, temporary / "raw" / name)
        write_json(temporary / "natural_terminal_gate.json", result["terminal"])
        old_output = tied_head.OUTPUT
        tied_head.OUTPUT = temporary
        try:
            _, fresh = tied_head.oracle_comparison(
                hidden,
                model,
                result["terminal"],
                bits,
                accumulators,
                top_k,
            )
        finally:
            tied_head.OUTPUT = old_output
    stored = result["comparison"]
    for key in (
        "status",
        "records_compared",
        "logit_bit_mismatch_count",
        "accumulator_mismatch_count",
        "top_k_mismatch_count",
        "selected_token_id",
        "selected_logit_f16_bits",
        "selected_token_logit_match",
        "failure_count",
        "first_material_mismatch",
    ):
        require(
            fresh[key] == stored[key],
            f"fresh tied-head semantic comparison drift: {key}",
        )


def validate(attempt_root: Path) -> dict[str, Any]:
    document = load_json(attempt_root / EVIDENCE_NAME)
    _validate_core_contract(document)
    require(
        authenticate_host_predecessor() == document["reviewed_predecessor"],
        "reviewed predecessor changed",
    )
    for record in document["source_bindings"]:
        _validate_file_record(record, "source")
    with configured_transaction_engine():
        validated = transaction_engine.validate_live_traversal(
            document["decoder_traversal"],
            token_embedding(INJECTED_TOKEN_ID),
        )
    require(
        cache_lineage(validated) == document["cache_lineage"],
        "fresh persistent K/V lineage comparison failed",
    )
    validate_final_rmsnorm(document)
    validate_tied_head(document)
    require(
        independent_policy_assessment(
            document["final_rmsnorm"],
            document["tied_lm_head_topk"],
        )
        == document["independent_policy_assessment"],
        "independent FP16 policy assessment changed",
    )
    require(
        focused_drift_checks(document) == document["focused_drift_validation"],
        "focused drift validation record mismatch",
    )
    seal = load_json(attempt_root / SEAL_NAME)
    require(
        seal.get("kind") == "ace3_model24_persistent_kv_second_token_seal"
        and seal.get("attempt_id") == attempt_root.name
        and seal.get("status") == "ENGINEERING_RESULT_REVIEW_REQUIRED",
        "persistent-K/V seal identity mismatch",
    )
    actual = {
        record["relative_path"]: record
        for record in snapshot(attempt_root)
    }
    expected = {
        record["relative_path"]: record
        for record in seal.get("artifacts", [])
    }
    require(actual == expected, "sealed persistent-K/V artifact closure changed")
    return document


def print_summary(document: Mapping[str, Any]) -> None:
    print(
        "MODEL24_PERSISTENT_KV_SECOND_TOKEN_REVIEW_REQUIRED "
        f"attempt={document['attempt_id']} history={document['input']['token_history']} "
        "layers=24 natural_terminals=24 cache_updates=24 "
        "local_exact_oracle_mismatches=0 independent_policy=NUMERIC_MISMATCH "
        f"independent_final_norm_failures="
        f"{document['independent_policy_assessment']['terminal_final_rmsnorm']['failure_count']} "
        "final_rmsnorm_local_oracle=PASS tied_head_local_oracle=PASS "
        f"selected_token={document['tied_lm_head_topk']['selected_token_id']} "
        f"selected_logit={document['tied_lm_head_topk']['selected_logit_f16_bits']} "
        f"decoded={document['host_continuation']['decoded_transcript']!r} "
        "synthesis=not_run ppa=not_measured fpga=not_run"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("generate", "validate"))
    parser.add_argument("--attempt-dir", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument("--completed-decoder-attempt", type=Path)
    args = parser.parse_args()
    attempt_root = args.attempt_dir.resolve()
    try:
        if args.operation == "generate":
            document = generate(
                attempt_root,
                args.completed_decoder_attempt,
            )
        else:
            require(
                args.completed_decoder_attempt is None,
                "--completed-decoder-attempt is only valid with generate",
            )
            document = validate(attempt_root)
    except Exception as error:
        if args.operation == "generate" and attempt_root.is_dir():
            failure = {
                "schema_version": 1,
                "status": "FAIL",
                "attempt_id": attempt_root.name,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failure_path = attempt_root / "failure.json"
            if not failure_path.exists():
                write_json(failure_path, failure)
        print(
            f"MODEL24_PERSISTENT_KV_SECOND_TOKEN_FAIL "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print_summary(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
