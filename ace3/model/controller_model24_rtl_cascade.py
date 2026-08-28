#!/usr/bin/env python3
"""Run the checkpoint controller's 24 launches through compiled decoder RTL."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors import safe_open

from controller_model24_cascade import (
    _canonical_json,
    _check_tensor_against_binding,
    _load_json,
    _load_raw_hidden,
    _sha256,
    parse_controller_events,
    parse_simulation_terminal,
    validate_binding_document,
)
from model24_execution_oracle import materialize_indexed_decoder_vectors
from model24_oracle import authenticate_checkpoint
from official_model24_next_token import (
    HIDDEN_SIZE,
    LAYER_COUNT,
    TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
    _bits_to_f16,
    _canonical_bytes,
    _f16_to_bits,
    _load_embeddings,
    _reference_layer,
)

KIND = "ace3_controller_model24_rtl_cascade"
TERMINAL_SCHEMA = "ace3_controller_model24_rtl_cascade_v1"
FP16_RELATIVE_TOLERANCE = 0.001
FP16_MAX_ULP_DISTANCE = 1
FRESH_EXECUTION_PATHS = (
    "initial_embeddings.hex",
    "layers",
    "compiled",
    "execution.json",
    "manifest.json",
    "terminal.txt",
)


class RtlCascadeError(RuntimeError):
    """Raised when resumable RTL evidence fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RtlCascadeError(message)


def hash_file(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": _sha256(payload)}


def prepare_fresh_output(output_dir: Path) -> dict[str, Any]:
    layer8_paths = {
        "layers/layer08": output_dir / "layers" / "layer08",
        "compiled/layer8": output_dir / "compiled" / "layer8",
    }
    layer8_before = {
        name: path.exists() for name, path in layer8_paths.items()
    }
    removed = []
    for relative in FRESH_EXECUTION_PATHS:
        path = output_dir / relative
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(relative)
    layer8_after = {
        name: path.exists() for name, path in layer8_paths.items()
    }
    require(
        not any(layer8_after.values()),
        "stale layer-8 execution scratch remains after fresh cleanup",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "kind": "ace3_controller_model24_rtl_cascade_fresh_preflight",
        "execution_scratch_paths": list(FRESH_EXECUTION_PATHS),
        "removed_existing_paths": removed,
        "layer8_execution_scratch_before": layer8_before,
        "layer8_execution_scratch_after": layer8_after,
        "stale_layer8_reused": False,
    }
    (output_dir / "fresh_preflight.json").write_bytes(
        _canonical_json(record)
    )
    return record


def raw_hidden_payload(hidden: np.ndarray) -> bytes:
    return "".join(
        f"{token:02x}{index:04x}{int(value):04x}\n"
        for token, row in enumerate(hidden)
        for index, value in enumerate(row)
    ).encode("ascii")


def _ordered_fp16_bits(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint16)
    magnitude = (values & np.uint16(0x7FFF)).astype(np.int32)
    negative = (values & np.uint16(0x8000)) != 0
    return np.where(negative, 0x8000 - magnitude, 0x8000 + magnitude)


def scale_aware_fp16_hidden_comparison(
    produced_bits: np.ndarray,
    reference_values: np.ndarray,
) -> dict[str, Any]:
    produced_bits = np.asarray(produced_bits, dtype=np.uint16)
    produced_values = _bits_to_f16(produced_bits).astype(np.float64)
    reference_values = np.asarray(reference_values, dtype=np.float64)
    require(
        produced_values.shape == reference_values.shape == (2, HIDDEN_SIZE),
        "two-token hidden comparison shape mismatch",
    )
    require(
        np.all(np.isfinite(produced_values)),
        "produced hidden state contains non-finite FP16 values",
    )
    require(
        np.all(np.isfinite(reference_values)),
        "reference hidden state contains non-finite values",
    )
    rounded_reference = reference_values.astype(np.float16)
    require(
        np.all(np.isfinite(rounded_reference)),
        "reference hidden state is outside the finite FP16 range",
    )
    reference_bits = _f16_to_bits(rounded_reference)
    difference = np.abs(produced_values - reference_values)
    relative = difference / np.maximum(
        np.abs(reference_values),
        float(np.finfo(np.float16).tiny),
    )
    ulp_distance = np.abs(
        _ordered_fp16_bits(produced_bits)
        - _ordered_fp16_bits(reference_bits)
    )
    accepted_absolute = difference <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE
    accepted_relative_ulp = (
        (relative < FP16_RELATIVE_TOLERANCE)
        & (ulp_distance <= FP16_MAX_ULP_DISTANCE)
    )
    accepted = accepted_absolute | accepted_relative_ulp

    def token_record(token_index: int) -> dict[str, Any]:
        token_accepted_absolute = accepted_absolute[token_index]
        token_accepted_relative_ulp = (
            accepted_relative_ulp[token_index] & ~token_accepted_absolute
        )
        token_accepted = accepted[token_index]
        return {
            "token_index": token_index,
            "max_abs_error": float(difference[token_index].max()),
            "mean_abs_error": float(difference[token_index].mean()),
            "max_relative_error": float(relative[token_index].max()),
            "max_ulp_distance": int(ulp_distance[token_index].max()),
            "accepted_by_absolute": int(token_accepted_absolute.sum()),
            "accepted_by_relative_ulp": int(
                token_accepted_relative_ulp.sum()
            ),
            "failure_count": int((~token_accepted).sum()),
            "within_tolerance": bool(token_accepted.all()),
        }

    failed = np.argwhere(~accepted)
    first_failure = None
    if failed.size:
        token_index, hidden_index = (int(value) for value in failed[0])
        first_failure = {
            "token_index": token_index,
            "hidden_index": hidden_index,
            "produced_bits": f"{int(produced_bits[token_index, hidden_index]):04x}",
            "produced_value": float(produced_values[token_index, hidden_index]),
            "reference_value": float(reference_values[token_index, hidden_index]),
            "absolute_error": float(difference[token_index, hidden_index]),
            "relative_error": float(relative[token_index, hidden_index]),
            "ulp_distance": int(ulp_distance[token_index, hidden_index]),
        }
    relative_only = accepted_relative_ulp & ~accepted_absolute
    tokens = [token_record(token_index) for token_index in range(2)]
    return {
        "decision_rule": (
            "absolute_error <= absolute_tolerance OR "
            "(relative_error < relative_tolerance AND "
            "ulp_distance <= max_ulp_distance)"
        ),
        "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
        "relative_tolerance": FP16_RELATIVE_TOLERANCE,
        "max_ulp_distance_allowed": FP16_MAX_ULP_DISTANCE,
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "max_relative_error": float(relative.max()),
        "max_ulp_distance": int(ulp_distance.max()),
        "accepted_by_absolute": int(accepted_absolute.sum()),
        "accepted_by_relative_ulp": int(relative_only.sum()),
        "failure_count": int((~accepted).sum()),
        "first_failure": first_failure,
        "within_tolerance": bool(accepted.all()),
        "tokens": tokens,
    }


def same_handoff_reference_layer(
    layer_id: int,
    tensors: Mapping[str, np.ndarray],
    handoff_bits: np.ndarray,
) -> torch.Tensor:
    handoff_bits = np.asarray(handoff_bits, dtype=np.uint16)
    require(
        handoff_bits.shape == (2, HIDDEN_SIZE),
        "two-token FP16 handoff shape mismatch",
    )
    handoff_values = _bits_to_f16(handoff_bits).astype(np.float64)
    return _reference_layer(
        layer_id,
        tensors,
        torch.from_numpy(handoff_values),
    )


def layer_comparison_record(
    layer_id: int,
    produced_bits: np.ndarray,
    same_handoff_reference: np.ndarray,
    propagated_reference: np.ndarray,
) -> dict[str, Any]:
    local = scale_aware_fp16_hidden_comparison(
        produced_bits,
        same_handoff_reference,
    )
    accumulated = scale_aware_fp16_hidden_comparison(
        produced_bits,
        propagated_reference,
    )
    return {
        "schema_version": 4,
        "layer_index": layer_id,
        "independent_oracle": "PyTorch CPU float64 dequantized-AWQ",
        "reference_boundary": "same authenticated FP16 layer input handoff",
        **local,
        "accumulated_end_to_end_drift": {
            "reference_boundary": (
                "continuous float64 propagation from authenticated official "
                "FP16 embeddings"
            ),
            "acceptance_role": "reported only; not used by the layer gate",
            **accumulated,
        },
    }


def require_layer_comparison(
    layer_id: int,
    comparison: Mapping[str, Any],
) -> None:
    if comparison.get("within_tolerance") is True:
        return
    failure = comparison.get("first_failure")
    if isinstance(failure, Mapping):
        detail = (
            f"token {failure['token_index']} hidden {failure['hidden_index']} "
            f"abs={failure['absolute_error']} rel={failure['relative_error']} "
            f"ulp={failure['ulp_distance']}"
        )
    else:
        detail = "failure coordinates unavailable"
    raise RtlCascadeError(
        f"layer {layer_id} scale-aware FP16 comparison failed: {detail}"
    )


def require_integer_oracle_bit_exact(
    layer_id: int,
    rtl_trace: bytes,
    oracle_trace: bytes,
    rtl_final: bytes,
    oracle_final: bytes,
) -> None:
    require(
        rtl_trace == oracle_trace,
        f"layer {layer_id} RTL trace differs from independent integer oracle",
    )
    require(
        rtl_final == oracle_final,
        f"layer {layer_id} RTL final differs from independent integer oracle",
    )


def natural_terminal(layer_id: int) -> str:
    if layer_id == 0:
        prefix = "schema=ace3_decoder_layer0_raw_v1"
    else:
        prefix = f"schema=ace3_decoder_layer_raw_v1 layer_index={layer_id}"
    return (
        f"{prefix} natural_terminal=1 exit_code=0 trace_count=46676 "
        "final_count=1792 done_count=2\n"
    )


def source_hashes(repository_root: Path) -> dict[str, str]:
    paths = (
        "ace3/model/controller_model24_rtl_cascade.py",
        "ace3/model/model24_execution_oracle.py",
        "ace3/model/decoder_layer0_oracle.py",
        "ace3/model/fp16_adaptation_oracle.py",
        "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
        "ace3/rtl/ace3_fp16_silu_gate_core.sv",
        "ace3/tb/ace3_decoder_layer0_token_engine_main.cpp",
    )
    return {
        path: _sha256((repository_root / path).read_bytes())
        for path in paths
    }


def validate_completed_layer(
    layer_id: int,
    layer_dir: Path,
    expected_input_sha256: str,
    binding: Mapping[str, Any],
    expected_consumed_tensors: list[dict[str, Any]],
    sources: Mapping[str, str],
    *,
    checkpoint_path: Path,
    tensor_map_path: Path,
    predecessor_path: Path,
) -> dict[str, Any] | None:
    record_path = layer_dir / "record.json"
    if not record_path.exists():
        return None
    record = _load_json(record_path.read_bytes(), f"layer {layer_id} record")
    require(record.get("layer_index") == layer_id, "layer record index mismatch")
    require(record.get("namespace") == binding["namespace"],
            "layer record namespace mismatch")
    require(
        record.get("descriptor_sha256") == binding["descriptor_sha256"],
        "layer record descriptor mismatch",
    )
    require(
        record.get("input_raw_sha256") == expected_input_sha256,
        "layer record input lineage mismatch",
    )
    require(
        record.get("consumed_tensors") == expected_consumed_tensors,
        "layer record consumed tensor lineage mismatch",
    )
    require(record.get("sources") == sources, "layer record source mismatch")

    raw_dir = layer_dir / "raw"
    vector_dir = layer_dir / "vectors"
    manifest = _load_json(
        (vector_dir / "boundary_manifest.json").read_bytes(),
        f"layer {layer_id} integer oracle manifest",
    )
    expected_layer_binding = {
        key: binding[key]
        for key in ("layer_id", "namespace", "descriptor_sha256")
    }
    require(manifest.get("layer_index") == layer_id,
            "integer oracle layer index mismatch")
    require(manifest.get("layer_binding") == expected_layer_binding,
            "integer oracle layer binding mismatch")
    input_handoff = manifest.get("input_handoff", {})
    require(input_handoff.get("sha256") == expected_input_sha256,
            "integer oracle input handoff digest mismatch")
    require(input_handoff.get("source_layer_index") == (
                None if layer_id == 0 else layer_id - 1
            ),
            "integer oracle predecessor layer mismatch")
    require(input_handoff.get("consumer_layer_index") == layer_id,
            "integer oracle consumer layer mismatch")
    require(input_handoff.get("source") == (
                "authenticated official token embedding rows"
                if layer_id == 0
                else f"authenticated decoder layer {layer_id - 1} raw final rows"
            ),
            "integer oracle predecessor source mismatch")
    require(input_handoff.get("byte_preserved_as") == "inputs.hex",
            "integer oracle handoff preservation mismatch")
    require(
        _sha256((vector_dir / "inputs.hex").read_bytes())
        == expected_input_sha256,
        "integer oracle retained input bytes mismatch",
    )

    raw_trace = (raw_dir / "trace.hex").read_bytes()
    raw_final = (raw_dir / "final.hex").read_bytes()
    oracle_trace = (vector_dir / "trace.hex").read_bytes()
    oracle_final = (vector_dir / "final.hex").read_bytes()

    require(
        _sha256(predecessor_path.read_bytes()) == expected_input_sha256,
        f"layer {layer_id} actual predecessor bytes mismatch",
    )
    with tempfile.TemporaryDirectory(
        prefix=f"ace3-layer{layer_id:02d}-resume-oracle-"
    ) as temporary:
        regenerated_dir = Path(temporary) / "vectors"
        regenerated_manifest = materialize_indexed_decoder_vectors(
            checkpoint_path,
            tensor_map_path,
            predecessor_path,
            regenerated_dir,
            layer_index=layer_id,
            expected_handoff_sha256=expected_input_sha256,
            accurate_silu=True,
        )
        require(
            regenerated_manifest.get("layer_binding")
            == expected_layer_binding,
            f"layer {layer_id} regenerated integer oracle binding mismatch",
        )
        regenerated_handoff = regenerated_manifest.get("input_handoff", {})
        require(
            regenerated_handoff.get("sha256") == expected_input_sha256,
            f"layer {layer_id} regenerated integer oracle handoff mismatch",
        )
        regenerated_trace = (regenerated_dir / "trace.hex").read_bytes()
        regenerated_final = (regenerated_dir / "final.hex").read_bytes()

    require_integer_oracle_bit_exact(
        layer_id,
        raw_trace,
        regenerated_trace,
        raw_final,
        regenerated_final,
    )
    require(
        oracle_trace == regenerated_trace,
        f"layer {layer_id} retained integer oracle trace differs from "
        "independently regenerated trace",
    )
    require(
        oracle_final == regenerated_final,
        f"layer {layer_id} retained integer oracle final differs from "
        "independently regenerated final",
    )
    require(
        record.get("output_raw_sha256") == _sha256(raw_final),
        f"layer {layer_id} output digest does not match actual final.hex",
    )
    for name in ("trace.hex", "final.hex", "terminal.txt", "comparison.json"):
        require(
            hash_file(raw_dir / name) == record["artifacts"][name],
            f"layer {layer_id} {name} hash mismatch",
        )
    for name in ("boundary_manifest.json", "inputs.hex", "trace.hex", "final.hex"):
        require(
            hash_file(vector_dir / name) == record["oracle_artifacts"][name],
            f"layer {layer_id} integer oracle {name} hash mismatch",
        )
    require(
        _load_json(
            (raw_dir / "comparison.json").read_bytes(),
            f"layer {layer_id} comparison",
        ) == record.get("comparison"),
        f"layer {layer_id} comparison record mismatch",
    )
    require(
        (raw_dir / "terminal.txt").read_text(encoding="ascii")
        == natural_terminal(layer_id),
        f"layer {layer_id} did not reach a natural terminal",
    )
    return record


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    require(
        completed.returncode == 0,
        f"command failed with exit {completed.returncode}: {' '.join(command)}",
    )


def execute(
    repository_root: Path,
    checkpoint_path: Path,
    tensor_map_path: Path,
    bindings_path: Path,
    simulation_dir: Path,
    output_dir: Path,
    fresh: bool = False,
) -> dict[str, Any]:
    fresh_preflight = prepare_fresh_output(output_dir) if fresh else None
    parse_simulation_terminal(simulation_dir / "terminal.txt")
    layer_order = parse_controller_events(simulation_dir / "controller_events.hex")
    require(layer_order == list(range(LAYER_COUNT)), "controller launch order mismatch")
    bindings_payload = bindings_path.read_bytes()
    bindings = _load_json(bindings_payload, "layer bindings")
    validate_binding_document(bindings, repository_root, tensor_map_path.read_bytes())
    authenticate_checkpoint(checkpoint_path)
    sources = source_hashes(repository_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers_dir = output_dir / "layers"
    layers_dir.mkdir(exist_ok=True)
    initial_path = output_dir / "initial_embeddings.hex"

    torch.set_num_threads(1)
    with safe_open(checkpoint_path, framework="np") as checkpoint:
        embeddings, embedding_record = _load_embeddings(checkpoint)
        initial_bits = _f16_to_bits(embeddings)
        initial_payload = raw_hidden_payload(initial_bits)
        if initial_path.exists():
            require(
                initial_path.read_bytes() == initial_payload,
                "initial embedding handoff changed",
            )
        else:
            initial_path.write_bytes(initial_payload)
        predecessor_path = initial_path
        predecessor_sha256 = _sha256(initial_payload)
        propagated_reference_hidden = torch.from_numpy(
            embeddings.astype(np.float64)
        )
        records = []

        for layer_id in layer_order:
            binding = bindings["layers"][layer_id]
            tensors = {
                record["name"]: np.asarray(checkpoint.get_tensor(record["name"]))
                for record in binding["tensors"]
            }
            consumed = [
                _check_tensor_against_binding(record, tensors[record["name"]])
                for record in binding["tensors"]
            ]
            same_handoff_reference_hidden = same_handoff_reference_layer(
                layer_id,
                tensors,
                _load_raw_hidden(predecessor_path.read_bytes()),
            )
            propagated_reference_hidden = _reference_layer(
                layer_id,
                tensors,
                propagated_reference_hidden,
            )
            layer_dir = layers_dir / f"layer{layer_id:02d}"
            completed = validate_completed_layer(
                layer_id,
                layer_dir,
                predecessor_sha256,
                binding,
                consumed,
                sources,
                checkpoint_path=checkpoint_path,
                tensor_map_path=tensor_map_path,
                predecessor_path=predecessor_path,
            )
            if completed is None:
                if layer_dir.exists():
                    shutil.rmtree(layer_dir)
                vector_dir = layer_dir / "vectors"
                raw_dir = layer_dir / "raw"
                raw_dir.mkdir(parents=True)
                manifest = materialize_indexed_decoder_vectors(
                    checkpoint_path,
                    tensor_map_path,
                    predecessor_path,
                    vector_dir,
                    layer_index=layer_id,
                    expected_handoff_sha256=predecessor_sha256,
                    accurate_silu=True,
                )
                require(
                    manifest["layer_binding"] == {
                        key: binding[key]
                        for key in ("layer_id", "namespace", "descriptor_sha256")
                    },
                    f"layer {layer_id} materialized binding mismatch",
                )
                compile_log = layer_dir / "compile.log"
                run_command(
                    [
                        "make",
                        "--no-print-directory",
                        "model24-rtl-layer-compile",
                        f"MODEL24_RTL_LAYER_INDEX={layer_id}",
                        "MODEL24_RTL_ACCURATE_SILU=1",
                    ],
                    repository_root,
                    compile_log,
                )
                binary = (
                    output_dir
                    / "compiled"
                    / f"layer{layer_id}"
                    / "obj_dir"
                    / "Vace3_decoder_layer0_token_engine"
                )
                require(binary.is_file(), f"layer {layer_id} RTL binary missing")
                run_command(
                    [
                        str(binary),
                        "--layer-index",
                        str(layer_id),
                        "--vector-dir",
                        str(vector_dir),
                        "--raw-dir",
                        str(raw_dir),
                        "--progress-interval",
                        "1000000",
                    ],
                    repository_root,
                    layer_dir / "simulation.log",
                )
                require(
                    (raw_dir / "terminal.txt").read_text(encoding="ascii")
                    == natural_terminal(layer_id),
                    f"layer {layer_id} did not reach a natural terminal",
                )
                require_integer_oracle_bit_exact(
                    layer_id,
                    (raw_dir / "trace.hex").read_bytes(),
                    (vector_dir / "trace.hex").read_bytes(),
                    (raw_dir / "final.hex").read_bytes(),
                    (vector_dir / "final.hex").read_bytes(),
                )
                produced = _load_raw_hidden((raw_dir / "final.hex").read_bytes())
                comparison = layer_comparison_record(
                    layer_id,
                    produced,
                    same_handoff_reference_hidden.detach().cpu().numpy(),
                    propagated_reference_hidden.detach().cpu().numpy(),
                )
                (raw_dir / "comparison.json").write_bytes(
                    _canonical_json(comparison)
                )
                require_layer_comparison(layer_id, comparison)
                raw_sha256 = _sha256((raw_dir / "final.hex").read_bytes())
                completed = {
                    "schema_version": 2,
                    "layer_index": layer_id,
                    "namespace": binding["namespace"],
                    "descriptor_sha256": binding["descriptor_sha256"],
                    "input_raw_sha256": predecessor_sha256,
                    "output_raw_sha256": raw_sha256,
                    "consumed_tensors": consumed,
                    "numeric_profile": manifest["numeric_profile"],
                    "comparison": comparison,
                    "sources": sources,
                    "artifacts": {
                        name: hash_file(raw_dir / name)
                        for name in (
                            "trace.hex",
                            "final.hex",
                            "terminal.txt",
                            "comparison.json",
                        )
                    },
                    "oracle_artifacts": {
                        name: hash_file(vector_dir / name)
                        for name in (
                            "boundary_manifest.json",
                            "inputs.hex",
                            "trace.hex",
                            "final.hex",
                        )
                    },
                }
                (layer_dir / "record.json").write_bytes(
                    _canonical_json(completed)
                )
            else:
                produced = _load_raw_hidden(
                    (layer_dir / "raw" / "final.hex").read_bytes()
                )
                comparison = layer_comparison_record(
                    layer_id,
                    produced,
                    same_handoff_reference_hidden.detach().cpu().numpy(),
                    propagated_reference_hidden.detach().cpu().numpy(),
                )
                require(
                    completed["comparison"] == comparison,
                    f"layer {layer_id} resumed comparison mismatch",
                )
                require_layer_comparison(layer_id, comparison)
            records.append(completed)
            predecessor_path = layer_dir / "raw" / "final.hex"
            predecessor_sha256 = completed["output_raw_sha256"]
            del tensors

    terminal_comparison = records[-1]["comparison"]
    terminal_accumulated_drift = terminal_comparison[
        "accumulated_end_to_end_drift"
    ]
    require(
        terminal_comparison["within_tolerance"],
        "post-layer-23 hidden tolerance exceeded: "
        + ", ".join(
            f"token {token['token_index']}={token['max_abs_error']}"
            for token in terminal_comparison["tokens"]
        ),
    )
    document = {
        "schema_version": 1,
        "kind": KIND,
        "model_binding": bindings["model_binding"],
        "controller": {
            "bindings_sha256": _sha256(bindings_payload),
            "events": layer_order,
            "events_sha256": hash_file(
                simulation_dir / "controller_events.hex"
            )["sha256"],
            "terminal_sha256": hash_file(simulation_dir / "terminal.txt")[
                "sha256"
            ],
        },
        "input": embedding_record,
        "layers": records,
        "fresh_run": (
            {
                "required": True,
                "preflight": fresh_preflight,
                "preflight_artifact": hash_file(
                    output_dir / "fresh_preflight.json"
                ),
            }
            if fresh_preflight is not None
            else {"required": False}
        ),
        "post_layer23": {
            "raw_sha256": records[-1]["output_raw_sha256"],
            "max_abs_error": terminal_comparison["max_abs_error"],
            "decision_token_max_abs_error": terminal_comparison["tokens"][1][
                "max_abs_error"
            ],
            "tokens": terminal_comparison["tokens"],
            "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
            "relative_tolerance": FP16_RELATIVE_TOLERANCE,
            "max_ulp_distance_allowed": FP16_MAX_ULP_DISTANCE,
            "max_relative_error": terminal_comparison["max_relative_error"],
            "max_ulp_distance": terminal_comparison["max_ulp_distance"],
            "within_tolerance": True,
            "accumulated_end_to_end_drift": terminal_accumulated_drift,
        },
        "claim_boundary": {
            "demonstrated": "compiled Verilator decoder RTL layers 0 through 23",
            "legacy_layer012": (
                "preserved by the separate default-profile regression"
            ),
            "tokenizer_dialogue": "not produced",
            "tied_lm_head": "not executed",
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
            "latency": "not measured",
            "throughput": "not measured",
        },
    }
    execution_payload = _canonical_json(document)
    (output_dir / "execution.json").write_bytes(execution_payload)
    manifest = {
        "schema_version": 1,
        "kind": "ace3_controller_model24_rtl_cascade_manifest",
        "execution": {
            "bytes": len(execution_payload),
            "sha256": _sha256(execution_payload),
        },
        "completed_layers": LAYER_COUNT,
        "consumed_tensors": sum(
            len(record["consumed_tensors"]) for record in records
        ),
        "terminal_layer": 23,
        "max_abs_error": terminal_comparison["max_abs_error"],
        "decision_token_max_abs_error": terminal_comparison["tokens"][1][
            "max_abs_error"
        ],
        "tokens": terminal_comparison["tokens"],
        "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
        "relative_tolerance": FP16_RELATIVE_TOLERANCE,
        "max_ulp_distance_allowed": FP16_MAX_ULP_DISTANCE,
        "max_relative_error": terminal_comparison["max_relative_error"],
        "max_ulp_distance": terminal_comparison["max_ulp_distance"],
        "within_tolerance": True,
        "accumulated_end_to_end_drift": terminal_accumulated_drift,
    }
    (output_dir / "manifest.json").write_bytes(_canonical_json(manifest))
    (output_dir / "terminal.txt").write_text(
        f"schema={TERMINAL_SCHEMA} natural_terminal=1 exit_code=0 "
        f"completed_layers={LAYER_COUNT} terminal_layer=23\n",
        encoding="ascii",
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--simulation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    document = execute(
        args.repository_root.resolve(strict=True),
        args.checkpoint.resolve(strict=True),
        args.tensor_map.resolve(strict=True),
        args.bindings.resolve(strict=True),
        args.simulation_dir.resolve(strict=True),
        args.output_dir.resolve(),
        fresh=args.fresh,
    )
    print(
        "MODEL24_CONTROLLER_RTL_CASCADE_PASS "
        f"layers={len(document['layers'])} terminal_layer=23 "
        f"max_abs_error={document['post_layer23']['max_abs_error']} "
        f"token0_max_abs_error="
        f"{document['post_layer23']['tokens'][0]['max_abs_error']} "
        f"token1_max_abs_error="
        f"{document['post_layer23']['tokens'][1]['max_abs_error']} "
        "tokenizer_dialogue=not_produced tied_lm_head=not_executed "
        "synthesis=not_run ppa=not_measured fpga=not_run "
        "latency=not_measured throughput=not_measured"
    )


if __name__ == "__main__":
    main()
