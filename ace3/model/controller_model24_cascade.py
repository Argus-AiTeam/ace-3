#!/usr/bin/env python3
"""Run the accepted decoder arithmetic in the order emitted by the RTL controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors import safe_open

from decoder_layer0_oracle import run_token as run_decoder_layer_token
from model24_execution_oracle import (
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    LAYER_DESCRIPTOR_SHA256,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    TENSOR_MAP_SHA256,
    indexed_layer_uses_accurate_silu,
    layer_bindings,
)
from model24_oracle import authenticate_checkpoint
from official_model24_next_token import (
    HIDDEN_SIZE,
    LAYER_COUNT,
    LAYER_TENSOR_COUNT,
    TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
    _bits_to_f16,
    _canonical_bytes,
    _f16_to_bits,
    _layer_tensor_names,
    _load_embeddings,
    _reference_layer,
    _tensor_record,
)

BINDING_KIND = "ace3_controller_model24_layer_bindings"
EXECUTION_KIND = "ace3_controller_model24_cascade_execution"
COMPARISON_KIND = "ace3_controller_model24_post_layer23_comparison"
SIMULATION_TERMINAL_SCHEMA = "ace3_model24_controller_raw_v1"
EXECUTION_TERMINAL_SCHEMA = "ace3_controller_model24_execution_v1"
ACCEPTED_LAYER012_RAW_SHA256 = (
    "22768ac6b337f920faac7de59b4eb43a203e1db45cdf688820fcbb35cdfe3446",
    "2324470c304f23a372378af6f9f65cc7a646fbaa614882c4ced44110b99dca85",
    "244c9d1d52923ecfff743c165da563468746f47557284865a4b22910a967c511",
)
SOURCE_PATHS = (
    "ace3/contracts/model24_layer_controller.json",
    "ace3/contracts/model24_tensor_map.json",
    "ace3/rtl/ace3_model24_layer_controller.sv",
    "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
    "ace3/rtl/ace3_fp16_silu_gate_core.sv",
    "ace3/tb/ace3_model24_layer_controller_tb.sv",
    "ace3/tb/ace3_model24_layer_controller_main.cpp",
    "ace3/tb/ace3_decoder_layer0_token_engine_main.cpp",
    "ace3/model/controller_model24_cascade.py",
    "ace3/model/controller_model24_rtl_cascade.py",
    "ace3/model/official_model24_next_token.py",
    "ace3/model/official_single_decoder_layer.py",
    "ace3/model/model24_execution_oracle.py",
    "ace3/model/model24_oracle.py",
    "ace3/model/attention_oracle.py",
    "ace3/model/fp16_adaptation_oracle.py",
    "ace3/model/projection_oracle.py",
)


class ControllerCascadeError(RuntimeError):
    """Raised when controller or layer evidence fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ControllerCascadeError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _load_json(payload: bytes, source: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"{source} has duplicate key {key}")
            result[key] = value
        return result

    try:
        document = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControllerCascadeError(f"{source} is not valid JSON: {error}") from error
    _require(isinstance(document, dict), f"{source} root must be an object")
    return document


def _tensor_map(payload: bytes) -> dict[str, Any]:
    _require(_sha256(payload) == TENSOR_MAP_SHA256, "tensor map SHA256 mismatch")
    document = _load_json(payload, "tensor map")
    _require(document.get("kind") == "ace3_model24_official_tensor_address_map",
             "tensor map kind mismatch")
    _require(document.get("inventory", {}).get("layer_namespace_count") == LAYER_COUNT,
             "tensor map layer count mismatch")
    return document


def _source_records(repository_root: Path) -> list[dict[str, Any]]:
    records = []
    for relative in SOURCE_PATHS:
        payload = (repository_root / relative).read_bytes()
        records.append(
            {"path": relative, "bytes": len(payload), "sha256": _sha256(payload)}
        )
    return records


def _binding_layers(tensor_map: Mapping[str, Any]) -> list[dict[str, Any]]:
    namespaces = tensor_map.get("layer_namespaces")
    tensors = tensor_map.get("tensors")
    _require(isinstance(namespaces, list), "tensor map namespaces are missing")
    _require(isinstance(tensors, list), "tensor map tensors are missing")
    expected_bindings = layer_bindings()
    layers = []
    for layer_id in range(LAYER_COUNT):
        namespace = f"model.layers.{layer_id}."
        namespace_records = [
            item for item in namespaces
            if isinstance(item, dict) and item.get("layer_id") == layer_id
        ]
        _require(len(namespace_records) == 1,
                 f"layer {layer_id} namespace is not unique")
        expected_binding = expected_bindings[layer_id]
        _require(namespace_records[0] == {
            "layer_id": layer_id,
            "namespace": namespace,
            "tensor_count": LAYER_TENSOR_COUNT,
            "descriptor_sha256": LAYER_DESCRIPTOR_SHA256[layer_id],
        }, f"layer {layer_id} namespace binding mismatch")
        records = sorted(
            (
                dict(item) for item in tensors
                if isinstance(item, dict)
                and isinstance(item.get("name"), str)
                and item["name"].startswith(namespace)
            ),
            key=lambda item: item["name"],
        )
        _require(
            [item["name"] for item in records] == list(_layer_tensor_names(layer_id)),
            f"layer {layer_id} tensor inventory mismatch",
        )
        layers.append(
            {
                **expected_binding,
                "tensor_count": len(records),
                "tensors": records,
            }
        )
    return layers


def build_binding_document(
    repository_root: Path,
    tensor_map_payload: bytes,
) -> dict[str, Any]:
    tensor_map = _tensor_map(tensor_map_payload)
    return {
        "schema_version": 1,
        "kind": BINDING_KIND,
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_bytes": CHECKPOINT_SIZE,
            "tensor_map_sha256": TENSOR_MAP_SHA256,
        },
        "controller": {
            "layer_count": LAYER_COUNT,
            "strict_layer_order": list(range(LAYER_COUNT)),
            "checkpoint_gated": True,
        },
        "sources": _source_records(repository_root),
        "layers": _binding_layers(tensor_map),
    }


def validate_binding_document(
    document: Mapping[str, Any],
    repository_root: Path,
    tensor_map_payload: bytes,
) -> None:
    expected = build_binding_document(repository_root, tensor_map_payload)
    _require(document == expected, "layer binding manifest mismatch")


def generate_bindings(
    repository_root: Path,
    checkpoint_path: Path,
    tensor_map_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    authenticate_checkpoint(checkpoint_path)
    document = build_binding_document(repository_root, tensor_map_path.read_bytes())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_json(document))
    return document


def parse_simulation_terminal(path: Path) -> dict[str, int]:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise ControllerCascadeError(f"simulation terminal unavailable: {error}") from error
    pattern = re.compile(
        rf"schema={SIMULATION_TERMINAL_SCHEMA} natural_terminal=(?P<natural>[01]) "
        r"exit_code=(?P<exit>[0-9]+) launches=(?P<launches>[0-9]+) "
        r"checkpoints=(?P<checkpoints>[0-9]+) done=(?P<done>[0-9]+) "
        r"terminal_layer=(?P<terminal>none|[0-9]+)\n"
    )
    match = pattern.fullmatch(text)
    _require(match is not None, "simulation terminal is malformed or ambiguous")
    values = {
        key: (-1 if value == "none" else int(value))
        for key, value in match.groupdict().items()
    }
    _require(
        values == {
            "natural": 1,
            "exit": 0,
            "launches": 24,
            "checkpoints": 24,
            "done": 1,
            "terminal": 23,
        },
        "simulation did not reach the natural post-layer-23 terminal",
    )
    return values


def parse_controller_events(path: Path) -> list[int]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ControllerCascadeError(f"controller event trace unavailable: {error}") from error
    _require(len(lines) == 49, "controller event trace must contain 49 events")
    words = []
    for ordinal, line in enumerate(lines):
        _require(re.fullmatch(r"[0-9a-f]{8}", line) is not None,
                 f"controller event {ordinal} is malformed")
        words.append(int(line, 16))
    expected = []
    for layer_id in range(LAYER_COUNT):
        expected.extend(
            (
                0x10000000 | layer_id,
                0x20000000
                | ((1 if layer_id == 23 else 0) << 10)
                | ((layer_id + 1) << 5)
                | layer_id,
            )
        )
    expected.append(0x30000000 | 23)
    _require(words == expected, "controller trace violates checkpoint-gated layer order")
    return [words[index] & 0x1F for index in range(0, 48, 2)]


def _execution_terminal(
    natural: int,
    exit_code: int,
    completed_layers: int,
    final_records: int,
) -> bytes:
    return (
        f"schema={EXECUTION_TERMINAL_SCHEMA} natural_terminal={natural} "
        f"exit_code={exit_code} completed_layers={completed_layers} "
        f"final_records={final_records}\n"
    ).encode("ascii")


def _parse_execution_terminal(path: Path) -> dict[str, int]:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise ControllerCascadeError(f"execution terminal unavailable: {error}") from error
    match = re.fullmatch(
        rf"schema={EXECUTION_TERMINAL_SCHEMA} natural_terminal=(?P<natural>[01]) "
        r"exit_code=(?P<exit>[0-9]+) completed_layers=(?P<layers>[0-9]+) "
        r"final_records=(?P<records>[0-9]+)\n",
        text,
    )
    _require(match is not None, "execution terminal is malformed or ambiguous")
    values = {key: int(value) for key, value in match.groupdict().items()}
    _require(values == {"natural": 1, "exit": 0, "layers": 24, "records": 1792},
             "execution did not naturally complete layer 23")
    return values


def _raw_hidden_payload(hidden: np.ndarray) -> bytes:
    return "".join(
        f"{token:02x}{index:04x}{int(value):04x}\n"
        for token, row in enumerate(hidden)
        for index, value in enumerate(row)
    ).encode("ascii")


def _load_raw_hidden(payload: bytes) -> np.ndarray:
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ControllerCascadeError(f"raw hidden state is not ASCII: {error}") from error
    _require(len(lines) == 2 * HIDDEN_SIZE, "raw hidden-state record count mismatch")
    hidden = np.zeros((2, HIDDEN_SIZE), dtype="<u2")
    for ordinal, line in enumerate(lines):
        _require(re.fullmatch(r"[0-9a-f]{10}", line) is not None,
                 f"raw hidden-state record {ordinal} is malformed")
        token = int(line[0:2], 16)
        index = int(line[2:6], 16)
        _require((token, index) == divmod(ordinal, HIDDEN_SIZE),
                 f"raw hidden-state record {ordinal} is out of order")
        hidden[token, index] = int(line[6:10], 16)
    return hidden


def _check_tensor_against_binding(
    record: Mapping[str, Any],
    tensor: np.ndarray,
) -> dict[str, Any]:
    actual = _tensor_record(str(record["name"]), tensor)
    _require(
        actual["dtype"] == record["dtype"]
        and actual["shape"] == record["shape"]
        and actual["bytes"] == record["byte_length"],
        f"{record['name']} checkpoint metadata mismatch",
    )
    return actual


def _hidden_comparison(
    produced_bits: np.ndarray,
    reference_values: np.ndarray,
) -> dict[str, Any]:
    produced_values = _bits_to_f16(produced_bits).astype(np.float64)
    reference_values = np.asarray(reference_values, dtype=np.float64)
    _require(
        produced_values.shape == reference_values.shape == (2, HIDDEN_SIZE),
        "two-token hidden comparison shape mismatch",
    )
    difference = np.abs(produced_values - reference_values)
    tokens = [
        {
            "token_index": token_index,
            "max_abs_error": float(token_difference.max()),
            "mean_abs_error": float(token_difference.mean()),
            "within_tolerance": bool(
                token_difference.max() <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE
            ),
        }
        for token_index, token_difference in enumerate(difference)
    ]
    return {
        "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "within_tolerance": all(token["within_tolerance"] for token in tokens),
        "tokens": tokens,
    }


def _run_accepted_decoder_layer(
    layer_id: int,
    binding: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray],
    hidden: np.ndarray,
) -> tuple[np.ndarray, int]:
    namespace = str(binding["namespace"])
    values: dict[str, list[int]] = {}
    for record in binding["tensors"]:
        name = str(record["name"])
        suffix = name.removeprefix(namespace)
        dtype = "<u2" if record["dtype"] == "F16" else "<u4"
        values[f"model.layers.0.{suffix}:"] = (
            np.ascontiguousarray(tensors[name]).view(dtype).reshape(-1)
            .astype(np.uint64)
            .tolist()
        )
    cache_k: list[list[int]] = []
    cache_v: list[list[int]] = []
    final_rows = []
    trace_records = 0
    for token, activation in enumerate(hidden):
        final, trace = run_decoder_layer_token(
            values,
            activation.tolist(),
            token,
            cache_k,
            cache_v,
            accurate_silu=indexed_layer_uses_accurate_silu(layer_id),
        )
        final_rows.append(final)
        trace_records += len(trace)
    result = np.asarray(final_rows, dtype="<u2")
    _require(
        result.shape == (2, HIDDEN_SIZE) and trace_records == 46676,
        f"layer {layer_id} accepted decoder output is incomplete",
    )
    return result, trace_records


def execute_cascade(
    repository_root: Path,
    checkpoint_path: Path,
    tensor_map_path: Path,
    bindings_path: Path,
    simulation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    parse_simulation_terminal(simulation_dir / "terminal.txt")
    layer_order = parse_controller_events(simulation_dir / "controller_events.hex")
    tensor_map_payload = tensor_map_path.read_bytes()
    bindings_payload = bindings_path.read_bytes()
    bindings = _load_json(bindings_payload, "layer bindings")
    validate_binding_document(bindings, repository_root, tensor_map_payload)
    authenticate_checkpoint(checkpoint_path)
    torch.set_num_threads(1)
    layer_records = []
    completed_layers = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    with safe_open(checkpoint_path, framework="np") as checkpoint:
        embeddings, embedding_record = _load_embeddings(checkpoint)
        hidden = _f16_to_bits(embeddings)
        reference_hidden = torch.from_numpy(embeddings.astype(np.float64))
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
            input_sha256 = _sha256(_canonical_bytes(hidden))
            hidden, trace_records = _run_accepted_decoder_layer(
                layer_id,
                binding,
                tensors,
                hidden,
            )
            reference_hidden = _reference_layer(
                layer_id,
                tensors,
                reference_hidden,
            )
            comparison = _hidden_comparison(
                hidden,
                reference_hidden.detach().cpu().numpy(),
            )
            output_sha256 = _sha256(_canonical_bytes(hidden))
            raw_output_sha256 = _sha256(_raw_hidden_payload(hidden))
            if layer_id < len(ACCEPTED_LAYER012_RAW_SHA256):
                _require(
                    raw_output_sha256 == ACCEPTED_LAYER012_RAW_SHA256[layer_id],
                    f"layer {layer_id} output regressed from accepted cascade",
                )
            layer_records.append(
                {
                    "layer_index": layer_id,
                    "namespace": binding["namespace"],
                    "descriptor_sha256": binding["descriptor_sha256"],
                    "input_hidden_sha256": input_sha256,
                    "output_hidden_sha256": output_sha256,
                    "post_layer_hidden_sha256": output_sha256,
                    "raw_output_sha256": raw_output_sha256,
                    "accepted_layer012_match": (
                        raw_output_sha256 == ACCEPTED_LAYER012_RAW_SHA256[layer_id]
                        if layer_id < len(ACCEPTED_LAYER012_RAW_SHA256)
                        else None
                    ),
                    "consumed_tensors": consumed,
                    "decoder_trace_records": trace_records,
                    "numeric_profile": {
                        "silu": (
                            "exp range-reduced degree-7 Q24"
                            if indexed_layer_uses_accurate_silu(layer_id)
                            else "accepted rational Q24"
                        ),
                    },
                    "independent_reference": comparison,
                }
            )
            completed_layers += 1
            del tensors
    raw_payload = _raw_hidden_payload(hidden)
    raw_path = output_dir / "raw_post_layer23.hex"
    raw_path.write_bytes(raw_payload)
    trace_payload = (simulation_dir / "controller_events.hex").read_bytes()
    simulation_terminal_payload = (simulation_dir / "terminal.txt").read_bytes()
    document = {
        "schema_version": 1,
        "kind": EXECUTION_KIND,
        "model_binding": bindings["model_binding"],
        "controller_binding": {
            "bindings_sha256": _sha256(bindings_payload),
            "events_sha256": _sha256(trace_payload),
            "simulation_terminal_sha256": _sha256(simulation_terminal_payload),
            "launched_layers": layer_order,
        },
        "input": {
            "embedding": embedding_record,
            "token_count": 2,
            "hidden_size": HIDDEN_SIZE,
        },
        "layers": layer_records,
        "post_layer23_hidden": {
            "shape": [2, HIDDEN_SIZE],
            "dtype": "F16",
            "records": 2 * HIDDEN_SIZE,
            "binary_sha256": _sha256(_canonical_bytes(hidden)),
            "decision_token_index": 1,
            "decision_token_sha256": _sha256(_canonical_bytes(hidden[-1])),
            "raw_sha256": _sha256(raw_payload),
        },
        "claim_boundary": {
            "demonstrated": (
                "RTL-controller-driven software decoder cascade through post-layer-23"
            ),
            "tokenizer_dialogue": "not produced by this evidence",
            "tied_lm_head": "not executed by this evidence",
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
            "latency": "not measured",
            "throughput": "not measured",
        },
    }
    (output_dir / "execution.json").write_bytes(_canonical_json(document))
    (output_dir / "terminal.txt").write_bytes(
        _execution_terminal(1, 0, completed_layers, 2 * HIDDEN_SIZE)
    )
    return document


def compare_with_official_oracle(
    repository_root: Path,
    checkpoint_path: Path,
    tensor_map_path: Path,
    bindings_path: Path,
    simulation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    parse_simulation_terminal(simulation_dir / "terminal.txt")
    _parse_execution_terminal(output_dir / "terminal.txt")
    layer_order = parse_controller_events(simulation_dir / "controller_events.hex")
    bindings_payload = bindings_path.read_bytes()
    bindings = _load_json(bindings_payload, "layer bindings")
    validate_binding_document(bindings, repository_root, tensor_map_path.read_bytes())
    execution_payload = (output_dir / "execution.json").read_bytes()
    execution = _load_json(execution_payload, "cascade execution")
    raw_payload = (output_dir / "raw_post_layer23.hex").read_bytes()
    produced_bits = _load_raw_hidden(raw_payload)
    validate_execution_document(execution, bindings, produced_bits, raw_payload)
    authenticate_checkpoint(checkpoint_path)
    torch.set_num_threads(1)
    with safe_open(checkpoint_path, framework="np") as checkpoint:
        embeddings, _ = _load_embeddings(checkpoint)
        reference_hidden = torch.from_numpy(embeddings.astype(np.float64))
        for layer_id in layer_order:
            tensors = {
                record["name"]: np.asarray(checkpoint.get_tensor(record["name"]))
                for record in bindings["layers"][layer_id]["tensors"]
            }
            authenticated_records = [
                _check_tensor_against_binding(record, tensors[record["name"]])
                for record in bindings["layers"][layer_id]["tensors"]
            ]
            _require(
                authenticated_records
                == execution["layers"][layer_id]["consumed_tensors"],
                f"execution layer {layer_id} consumed tensor hash mismatch",
            )
            reference_hidden = _reference_layer(layer_id, tensors, reference_hidden)
            del tensors
    reference_values = reference_hidden.detach().cpu().numpy()
    result = _hidden_comparison(produced_bits, reference_values)
    _require(
        result["within_tolerance"],
        "post-layer-23 hidden tolerance exceeded: "
        + ", ".join(
            f"token {token['token_index']}={token['max_abs_error']}"
            for token in result["tokens"]
        ),
    )
    reference_payload = np.ascontiguousarray(
        reference_values,
        dtype="<f8",
    ).tobytes()
    comparison = {
        "schema_version": 1,
        "kind": COMPARISON_KIND,
        "source_execution_sha256": _sha256(execution_payload),
        "produced": {
            "source": "post-layer-23 two-token hidden state",
            "token_indices": [0, 1],
            "shape": [2, HIDDEN_SIZE],
            "dtype": "F16",
            "sha256": _sha256(_canonical_bytes(produced_bits)),
        },
        "official_oracle": {
            "implementation": "PyTorch CPU float64 dequantized-AWQ Qwen2",
            "source": "post-layer-23 two-token hidden state",
            "token_indices": [0, 1],
            "shape": [2, HIDDEN_SIZE],
            "dtype": "F64",
            "sha256": _sha256(reference_payload),
        },
        "comparison": result,
        "post_layer23_boundary": True,
    }
    comparison_payload = _canonical_json(comparison)
    (output_dir / "comparison.json").write_bytes(comparison_payload)
    artifacts = {}
    for name, payload in (
        ("bindings.json", bindings_payload),
        ("controller_events.hex", (simulation_dir / "controller_events.hex").read_bytes()),
        ("simulation_terminal.txt", (simulation_dir / "terminal.txt").read_bytes()),
        ("execution.json", execution_payload),
        ("execution_terminal.txt", (output_dir / "terminal.txt").read_bytes()),
        ("raw_post_layer23.hex", raw_payload),
        ("comparison.json", comparison_payload),
    ):
        artifacts[name] = {"bytes": len(payload), "sha256": _sha256(payload)}
    manifest = {
        "schema_version": 1,
        "kind": "ace3_controller_model24_cascade_manifest",
        "artifacts": artifacts,
        "summary": {
            "layers": 24,
            "checkpoints": 24,
            "consumed_tensors": 24 * 26,
            "terminal_layer": 23,
            "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
            "max_abs_error": result["max_abs_error"],
            "tokens": result["tokens"],
            "within_tolerance": True,
        },
    }
    (output_dir / "manifest.json").write_bytes(_canonical_json(manifest))
    return comparison


def validate_execution_document(
    document: Mapping[str, Any],
    bindings: Mapping[str, Any],
    hidden: np.ndarray,
    raw_payload: bytes,
) -> None:
    _require(document.get("schema_version") == 1, "execution schema mismatch")
    _require(document.get("kind") == EXECUTION_KIND, "execution kind mismatch")
    _require(document.get("model_binding") == bindings["model_binding"],
             "execution model binding mismatch")
    layers = document.get("layers")
    _require(isinstance(layers, list) and len(layers) == LAYER_COUNT,
             "execution must contain 24 layers")
    previous_output = None
    for layer_id, layer in enumerate(layers):
        binding = bindings["layers"][layer_id]
        _require(
            layer["layer_index"] == layer_id
            and layer["namespace"] == binding["namespace"]
            and layer["descriptor_sha256"] == binding["descriptor_sha256"],
            f"execution layer {layer_id} binding mismatch",
        )
        if previous_output is not None:
            _require(layer["input_hidden_sha256"] == previous_output,
                     f"execution layer {layer_id} lineage mismatch")
        previous_output = layer["output_hidden_sha256"]
        _require(layer["post_layer_hidden_sha256"] == previous_output,
                 f"execution layer {layer_id} output mismatch")
        if layer_id < len(ACCEPTED_LAYER012_RAW_SHA256):
            _require(
                layer["raw_output_sha256"]
                == ACCEPTED_LAYER012_RAW_SHA256[layer_id]
                and layer["accepted_layer012_match"] is True,
                f"execution layer {layer_id} accepted continuity mismatch",
            )
        consumed = layer["consumed_tensors"]
        _require(len(consumed) == LAYER_TENSOR_COUNT,
                 f"execution layer {layer_id} tensor count mismatch")
        _require(
            [record["name"] for record in consumed]
            == [record["name"] for record in binding["tensors"]],
            f"execution layer {layer_id} tensor inventory mismatch",
        )
        _require(
            all(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) for record in consumed),
            f"execution layer {layer_id} tensor hash mismatch",
        )
    terminal = document["post_layer23_hidden"]
    _require(
        terminal == {
            "shape": [2, HIDDEN_SIZE],
            "dtype": "F16",
            "records": 2 * HIDDEN_SIZE,
            "binary_sha256": _sha256(_canonical_bytes(hidden)),
            "decision_token_index": 1,
            "decision_token_sha256": _sha256(_canonical_bytes(hidden[-1])),
            "raw_sha256": _sha256(raw_payload),
        },
        "execution post-layer-23 binding mismatch",
    )
    _require(previous_output == terminal["binary_sha256"],
             "execution terminal does not descend from layer 23")


def validate_evidence(
    repository_root: Path,
    tensor_map_path: Path,
    bindings_path: Path,
    simulation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    parse_simulation_terminal(simulation_dir / "terminal.txt")
    parse_controller_events(simulation_dir / "controller_events.hex")
    _parse_execution_terminal(output_dir / "terminal.txt")
    bindings_payload = bindings_path.read_bytes()
    bindings = _load_json(bindings_payload, "layer bindings")
    validate_binding_document(bindings, repository_root, tensor_map_path.read_bytes())
    execution_payload = (output_dir / "execution.json").read_bytes()
    execution = _load_json(execution_payload, "cascade execution")
    raw_payload = (output_dir / "raw_post_layer23.hex").read_bytes()
    validate_execution_document(
        execution,
        bindings,
        _load_raw_hidden(raw_payload),
        raw_payload,
    )
    comparison_payload = (output_dir / "comparison.json").read_bytes()
    comparison = _load_json(comparison_payload, "comparison")
    _require(comparison.get("kind") == COMPARISON_KIND, "comparison kind mismatch")
    _require(
        comparison.get("source_execution_sha256") == _sha256(execution_payload)
        and comparison.get("post_layer23_boundary") is True,
        "comparison execution binding mismatch",
    )
    result = comparison["comparison"]
    _require(
        result["absolute_tolerance"] == TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE
        and result["within_tolerance"] is True
        and result["max_abs_error"] <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
        "post-layer-23 comparison failed",
    )
    manifest_payload = (output_dir / "manifest.json").read_bytes()
    manifest = _load_json(manifest_payload, "cascade manifest")
    expected_payloads = {
        "bindings.json": bindings_payload,
        "controller_events.hex": (simulation_dir / "controller_events.hex").read_bytes(),
        "simulation_terminal.txt": (simulation_dir / "terminal.txt").read_bytes(),
        "execution.json": execution_payload,
        "execution_terminal.txt": (output_dir / "terminal.txt").read_bytes(),
        "raw_post_layer23.hex": raw_payload,
        "comparison.json": comparison_payload,
    }
    _require(
        manifest.get("artifacts")
        == {
            name: {"bytes": len(payload), "sha256": _sha256(payload)}
            for name, payload in expected_payloads.items()
        },
        "cascade manifest artifact binding mismatch",
    )
    return manifest["summary"]


def _write_failed_terminal(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "terminal.txt").write_bytes(_execution_terminal(0, 2, 0, 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=("bindings", "execute", "compare", "validate"),
    )
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--simulation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve(strict=True)
    tensor_map = args.tensor_map.resolve(strict=True)
    bindings_path = args.bindings.resolve()
    checkpoint = args.checkpoint.resolve(strict=True) if args.checkpoint else None
    simulation_dir = (
        args.simulation_dir.resolve(strict=True) if args.simulation_dir else None
    )
    output_dir = args.output_dir.resolve() if args.output_dir else None
    try:
        if args.operation == "bindings":
            _require(checkpoint is not None, "--checkpoint is required")
            document = generate_bindings(
                root, checkpoint, tensor_map, bindings_path
            )
            print(
                "MODEL24_CONTROLLER_BINDINGS_PASS "
                f"layers={len(document['layers'])} tensors="
                f"{sum(layer['tensor_count'] for layer in document['layers'])} "
                f"sha256={_sha256(bindings_path.read_bytes())}"
            )
        elif args.operation == "execute":
            _require(checkpoint is not None, "--checkpoint is required")
            _require(simulation_dir is not None, "--simulation-dir is required")
            _require(output_dir is not None, "--output-dir is required")
            try:
                document = execute_cascade(
                    root,
                    checkpoint,
                    tensor_map,
                    bindings_path,
                    simulation_dir,
                    output_dir,
                )
            except (ControllerCascadeError, OSError, RuntimeError, ValueError):
                _write_failed_terminal(output_dir)
                raise
            print(
                "MODEL24_CONTROLLER_CASCADE_EXECUTION_PASS "
                f"layers={len(document['layers'])} "
                f"tensors={sum(len(layer['consumed_tensors']) for layer in document['layers'])} "
                f"post_layer23_sha256="
                f"{document['post_layer23_hidden']['binary_sha256']}"
            )
        elif args.operation == "compare":
            _require(checkpoint is not None, "--checkpoint is required")
            _require(simulation_dir is not None, "--simulation-dir is required")
            _require(output_dir is not None, "--output-dir is required")
            comparison = compare_with_official_oracle(
                root,
                checkpoint,
                tensor_map,
                bindings_path,
                simulation_dir,
                output_dir,
            )
            result = comparison["comparison"]
            print(
                "MODEL24_CONTROLLER_CASCADE_COMPARISON_PASS "
                f"terminal_layer=23 tolerance={result['absolute_tolerance']} "
                f"max_abs_error={result['max_abs_error']}"
            )
        else:
            _require(simulation_dir is not None, "--simulation-dir is required")
            _require(output_dir is not None, "--output-dir is required")
            summary = validate_evidence(
                root,
                tensor_map,
                bindings_path,
                simulation_dir,
                output_dir,
            )
            print(
                "MODEL24_CONTROLLER_CASCADE_VALIDATION_PASS "
                f"layers={summary['layers']} checkpoints={summary['checkpoints']} "
                f"tensors={summary['consumed_tensors']} terminal_layer=23 "
                f"tolerance={summary['absolute_tolerance']} "
                f"max_abs_error={summary['max_abs_error']}"
            )
    except (ControllerCascadeError, OSError, ValueError) as error:
        raise SystemExit(f"MODEL24_CONTROLLER_CASCADE_FAIL {error}") from error


if __name__ == "__main__":
    main()
