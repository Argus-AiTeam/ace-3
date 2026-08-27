#!/usr/bin/env python3
"""Import sealed position-0 semantic K/V and execute token 2114 at position 1."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from generation_feedback_compare import parse_raw
from model24_first_voice_hybrid import (
    BUILD_MANIFEST_KIND,
    HIDDEN_SIZE,
    _authenticate_tensor_map,
    _bits_to_f16,
    _comparison,
    _f16_to_bits,
    _load_model,
    _reference_layer_step,
    _run_transaction,
    _serialize_layer_tensors,
    _validate_retained_state_record,
    authenticate_build,
    canonical_json,
    compiled_binary_hashes,
    contract_binding,
    hash_file,
    load_json,
    state_tip_commitment,
    validate_state_envelope,
    write_json,
)
from model24_oracle import (
    CHECKPOINT_SHA256,
    MODEL_REPOSITORY,
    MODEL_REVISION,
)
from model24_execution_oracle import TENSOR_MAP_SHA256
from official_model24_next_token import TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE
from official_single_decoder_layer import HEAD_DIM, KEY_VALUE_HEADS

SCHEMA = "ace3-position1-model24-causal-traversal-v2"
PARENT_SCHEMA = "ace3-position1-parent-set-v1"
SELECTED_TOKEN = 2114
POSITION = 1
LAYER_COUNT = 24
SEMANTIC_KV_SCHEMA = "ace3-semantic-kv-preload-v1"


class TraversalError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TraversalError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    document = load_json(path, "position-1 traversal contract")
    require(document.get("contract_id") == SCHEMA, "contract schema mismatch")
    return document


def _load_parent_kv(
    path: Path,
    expected: Mapping[str, Any] | None = None,
    layer_index: int | None = None,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    rows: dict[int, list[bytes]] = {6: [], 7: []}
    values: dict[int, list[int]] = {6: [], 7: []}
    label = f"layer {layer_index} " if layer_index is not None else ""
    with gzip.open(path, "rb") as source:
        for raw in source:
            line = raw.rstrip(b"\n")
            require(len(line) == 16, "parent trace row malformed")
            position = int(line[2:6], 16)
            stage = int(line[6:8], 16)
            if stage in rows:
                ordinal = len(rows[6]) + len(rows[7])
                expected_stage = 6 if ordinal % 2 == 0 else 7
                expected_index = ordinal // 2
                index = int(line[8:12], 16)
                require(position == 0, f"{label}parent K/V trace position mismatch")
                require(
                    stage == expected_stage and index == expected_index,
                    f"{label}parent K/V trace index/order mismatch",
                )
                rows[stage].append(line + b"\n")
                values[stage].append(int(line[12:16], 16))
    elements = KEY_VALUE_HEADS * HEAD_DIM
    require(
        len(rows[6]) == elements and len(rows[7]) == elements,
        f"{label}parent K/V trace count mismatch",
    )
    record = {
        "k_sha256": sha256(b"".join(rows[6])),
        "v_sha256": sha256(b"".join(rows[7])),
        "elements_each": elements,
        "format": "FP16",
    }
    if expected is not None:
        require(record == expected, f"{label}substituted parent K/V")
    shape = (POSITION, KEY_VALUE_HEADS, HEAD_DIM)
    reference_k = torch.from_numpy(
        np.asarray(values[6], dtype="<u2").view("<f2").astype(np.float64).reshape(shape)
    )
    reference_v = torch.from_numpy(
        np.asarray(values[7], dtype="<u2").view("<f2").astype(np.float64).reshape(shape)
    )
    return record, reference_k, reference_v


def _trace_parent_kv(path: Path) -> dict[str, Any]:
    return _load_parent_kv(path)[0]


def _layer_reference_comparisons(
    reference_state: Any,
    contract_input: torch.Tensor,
    continuous_input: torch.Tensor,
    actual_input_bits: np.ndarray,
    output_bits: np.ndarray,
    reference_k: torch.Tensor,
    reference_v: torch.Tensor,
    position: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    reference_state.reference_k = reference_k.clone()
    reference_state.reference_v = reference_v.clone()
    contract_output_float64 = _reference_layer_step(
        reference_state, contract_input.unsqueeze(0), position
    )[0]
    contract_output = torch.from_numpy(
        np.asarray(
            contract_output_float64.detach().cpu().numpy(), dtype="<f2"
        ).astype(np.float64)
    )
    reference_state.reference_k = reference_k.clone()
    reference_state.reference_v = reference_v.clone()
    continuous_output = _reference_layer_step(
        reference_state, continuous_input.unsqueeze(0), position
    )[0]
    reference_state.reference_k = reference_k.clone()
    reference_state.reference_v = reference_v.clone()
    actual_input = torch.from_numpy(
        _bits_to_f16(actual_input_bits).astype(np.float64)
    )
    local_output = _reference_layer_step(
        reference_state, actual_input.unsqueeze(0), position
    )[0]
    return (
        contract_output,
        continuous_output,
        local_output,
        _comparison(output_bits, contract_output),
        _comparison(output_bits, continuous_output),
        _comparison(output_bits, local_output),
    )


def _parse_hidden_words(path: Path, label: str) -> np.ndarray:
    lines = path.read_text(encoding="ascii").splitlines()
    require(len(lines) == HIDDEN_SIZE, f"{label} record count mismatch")
    values = np.empty(HIDDEN_SIZE, dtype="<u2")
    for expected_index, line in enumerate(lines):
        require(
            len(line) == 10
            and line[:2] == "00"
            and int(line[2:6], 16) == expected_index,
            f"{label} ordering mismatch",
        )
        values[expected_index] = int(line[6:10], 16)
    return values


def parse_semantic_kv_payload(path: Path, layer_index: int) -> dict[str, Any]:
    lines = path.read_text(encoding="ascii").splitlines()
    require(len(lines) == 256, "semantic K/V record count mismatch")
    source_rows: dict[int, list[bytes]] = {6: [], 7: []}
    for ordinal, line in enumerate(lines):
        require(len(line) == 18, "semantic K/V record width mismatch")
        try:
            layer = int(line[0:2], 16)
            slot = int(line[2:4], 16)
            position = int(line[4:8], 16)
            stage = int(line[8:10], 16)
            index = int(line[10:14], 16)
            value = int(line[14:18], 16)
        except ValueError as error:
            raise TraversalError("semantic K/V record is not hexadecimal") from error
        require(layer == layer_index, "semantic K/V cross-layer mismatch")
        require(slot == 0, "semantic K/V cache slot mismatch")
        require(position == 0, "semantic K/V source position mismatch")
        require(
            stage == (6 if ordinal % 2 == 0 else 7)
            and index == ordinal // 2,
            "semantic K/V reordered or duplicated record",
        )
        source_rows[stage].append(
            f"00{position:04x}{stage:02x}{index:04x}{value:04x}\n".encode("ascii")
        )
    return {
        "k_sha256": sha256(b"".join(source_rows[6])),
        "v_sha256": sha256(b"".join(source_rows[7])),
        "elements_each": 128,
        "format": "FP16",
    }


def write_semantic_kv_preload(
    trace_path: Path,
    destination: Path,
    *,
    layer_index: int,
    parent: Mapping[str, Any],
    parent_document: Mapping[str, Any],
    parent_set_sha256: str,
) -> tuple[Path, Path]:
    _load_parent_kv(trace_path, parent["parent_kv"], layer_index)
    payload_path = destination / f"layer{layer_index:02d}.hex"
    manifest_path = destination / f"layer{layer_index:02d}.json"
    require(not payload_path.exists() and not manifest_path.exists(), "semantic K/V preload already exists")
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    with gzip.open(trace_path, "rt", encoding="ascii", newline="") as source:
        for line in source:
            raw = line.rstrip("\n")
            stage = int(raw[6:8], 16)
            if stage in (6, 7):
                records.append(
                    f"{layer_index:02x}00{int(raw[2:6], 16):04x}{stage:02x}"
                    f"{int(raw[8:12], 16):04x}{int(raw[12:16], 16):04x}\n"
                )
    payload_path.write_text("".join(records), encoding="ascii", newline="")
    require(
        parse_semantic_kv_payload(payload_path, layer_index) == parent["parent_kv"],
        f"layer {layer_index} semantic K/V payload mismatch",
    )
    document = {
        "schema": SEMANTIC_KV_SCHEMA,
        "model_binding": parent_document["model_binding"],
        "parent_set_sha256": parent_set_sha256,
        "layer_index": layer_index,
        "cache_slot": 0,
        "source_position": 0,
        "execution_position": POSITION,
        "execution_token": SELECTED_TOKEN,
        "tensor_binding": {
            "key": "trace-stage-6-rotated-key-fp16",
            "value": "trace-stage-7-value-fp16",
            "ordering": "kv-head-major-dimension-minor",
        },
        "source_trace": parent["trace"],
        "parent_kv": parent["parent_kv"],
        "trusted_tip": parent["trusted_tip"],
        "payload": {"path": payload_path.name, **hash_file(payload_path)},
    }
    write_json(manifest_path, document)
    validate_semantic_kv_preload(
        manifest_path,
        payload_path,
        layer_index=layer_index,
        parent=parent,
        parent_document=parent_document,
        parent_set_sha256=parent_set_sha256,
    )
    return manifest_path, payload_path


def validate_semantic_kv_preload(
    manifest_path: Path,
    payload_path: Path,
    *,
    layer_index: int,
    parent: Mapping[str, Any],
    parent_document: Mapping[str, Any],
    parent_set_sha256: str,
) -> dict[str, Any]:
    document = load_json(manifest_path, "semantic K/V preload manifest")
    require(manifest_path.read_bytes() == canonical_json(document), "semantic K/V manifest is not canonical")
    require(
        document.get("schema") == SEMANTIC_KV_SCHEMA
        and document.get("model_binding") == parent_document["model_binding"]
        and document.get("parent_set_sha256") == parent_set_sha256
        and document.get("layer_index") == layer_index
        and document.get("cache_slot") == 0
        and document.get("source_position") == 0
        and document.get("execution_position") == POSITION
        and document.get("execution_token") == SELECTED_TOKEN
        and document.get("source_trace") == parent["trace"]
        and document.get("parent_kv") == parent["parent_kv"]
        and document.get("trusted_tip") == parent["trusted_tip"],
        f"layer {layer_index} semantic K/V binding mismatch",
    )
    payload = document.get("payload")
    require(
        isinstance(payload, Mapping)
        and payload.get("path") == payload_path.name
        and hash_file(payload_path) == {key: payload.get(key) for key in ("bytes", "sha256")},
        f"layer {layer_index} tampered semantic K/V payload",
    )
    require(
        parse_semantic_kv_payload(payload_path, layer_index) == parent["parent_kv"],
        f"layer {layer_index} substituted semantic K/V values",
    )
    return document


def _bound_file_matches(
    path: Path,
    record: Mapping[str, Any],
    expected_path: str,
) -> bool:
    return (
        path.is_file()
        and set(record) == {"path", "bytes", "sha256"}
        and record.get("path") == expected_path
        and hash_file(path)
        == {"bytes": record.get("bytes"), "sha256": record.get("sha256")}
    )


def authenticate_historical_build(
    compiled_dir: Path,
    contract_record: Mapping[str, Any],
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    manifest_path = compiled_dir / "build_manifest.json"
    require(manifest_path.is_file(), "historical build manifest is missing")
    manifest_payload = manifest_path.read_bytes()
    manifest = load_json(manifest_path, "historical build manifest")
    manifest_sha256 = sha256(manifest_payload)
    require(
        manifest_payload == canonical_json(manifest)
        and manifest_sha256 == expected_sha256
        and manifest.get("schema_version") == 2
        and manifest.get("kind") == BUILD_MANIFEST_KIND
        and manifest.get("verilator_savable") is True
        and manifest.get("compact_layout") is True
        and manifest.get("contract") == dict(contract_record),
        "historical build manifest binding mismatch",
    )
    sources = manifest.get("sources")
    layers = manifest.get("layers")
    require(
        isinstance(sources, Mapping)
        and isinstance(layers, list)
        and [record.get("layer_index") for record in layers]
        == list(range(LAYER_COUNT)),
        "historical build inventory mismatch",
    )
    for layer_index, record in enumerate(layers):
        manifest_record = record.get("manifest")
        binary_record = record.get("binary")
        layer_manifest_path = compiled_dir / f"layer{layer_index}/layer_manifest.json"
        binary_file = compiled_dir / f"layer{layer_index}/bin/Vace3_decoder_layer0_token_engine"
        require(
            isinstance(manifest_record, Mapping)
            and isinstance(binary_record, Mapping)
            and _bound_file_matches(
                layer_manifest_path,
                manifest_record,
                f"layer{layer_index}/layer_manifest.json",
            )
            and _bound_file_matches(
                binary_file,
                binary_record,
                f"layer{layer_index}/bin/Vace3_decoder_layer0_token_engine",
            ),
            f"historical layer {layer_index} file binding mismatch",
        )
        layer_payload = layer_manifest_path.read_bytes()
        layer_manifest = load_json(
            layer_manifest_path, f"historical layer {layer_index} manifest"
        )
        require(
            layer_payload == canonical_json(layer_manifest)
            and sha256(layer_payload) == record.get("manifest_sha256")
            and layer_manifest.get("schema_version") == 1
            and layer_manifest.get("kind")
            == "ace3_model24_first_voice_compact_layer"
            and layer_manifest.get("layer_index") == layer_index
            and layer_manifest.get("contract") == dict(contract_record)
            and layer_manifest.get("sources") == sources
            and layer_manifest.get("configuration_sha256")
            == record.get("configuration_sha256")
            and layer_manifest.get("binary") == dict(binary_record),
            f"historical layer {layer_index} manifest binding mismatch",
        )
    return manifest, manifest_sha256


def collect_parent_document(
    repository_root: Path,
    source_root: Path,
    compiled_dir: Path,
    expected_build_sha256: str,
) -> dict[str, Any]:
    _, first_voice_contract = contract_binding(repository_root)
    build_manifest, build_sha256 = authenticate_historical_build(
        compiled_dir, first_voice_contract, expected_build_sha256
    )
    binary_hashes = compiled_binary_hashes(build_manifest)
    layers = []
    previous_output = None
    for layer_index in range(LAYER_COUNT):
        state_path = source_root / f"states/layer{layer_index:02d}/position001/state"
        envelope_path = state_path.with_name("envelope.json")
        transaction_dir = source_root / f"transactions/position000/layer{layer_index:02d}"
        transaction_path = transaction_dir / "transaction.json"
        final_path = transaction_dir / "raw/final.hex"
        trace_path = transaction_dir / "raw/trace.hex.gz"
        terminal_path = transaction_dir / "raw/terminal.txt"
        input_path = transaction_dir / "inputs.hex"
        rope_path = transaction_dir / "rope_coefficients.hex"
        for path in (
            state_path, envelope_path, transaction_path, final_path,
            trace_path, terminal_path, input_path, rope_path,
        ):
            require(path.is_file(), f"missing parent artifact: {path}")
        envelope, envelope_payload = _validate_retained_state_record(
            states_dir=source_root / "states",
            runtime_dir=source_root / "transactions",
            layer_index=layer_index,
            next_position=POSITION,
            build_manifest_sha256=build_sha256,
            binary_sha256=binary_hashes[layer_index],
        )
        transaction = load_json(transaction_path, f"layer {layer_index} transaction")
        require(
            transaction.get("schema_version") == 1
            and transaction.get("kind") == "ace3_decoder_verilator_transaction"
            and transaction.get("layer_index") == layer_index
            and transaction.get("position") == 0
            and transaction.get("next_position") == 1
            and transaction.get("cache_slot") == 0
            and type(transaction.get("trace_records")) is int
            and transaction["trace_records"] > 0
            and transaction.get("final_records") == HIDDEN_SIZE
            and transaction.get("done_records") == 1
            and transaction.get("natural_terminal") is True,
            f"layer {layer_index} natural terminal mismatch",
        )
        if previous_output is not None:
            require(
                envelope["input_activation_sha256"] == previous_output,
                f"layer {layer_index} activation chain mismatch",
            )
        previous_output = envelope["output_hidden_sha256"]
        layers.append({
            "layer_index": layer_index,
            "binary_sha256": binary_hashes[layer_index],
            "state": hash_file(state_path),
            "envelope": hash_file(envelope_path),
            "transaction": hash_file(transaction_path),
            "inputs": hash_file(input_path),
            "final": hash_file(final_path),
            "trace": hash_file(trace_path),
            "terminal": hash_file(terminal_path),
            "rope": hash_file(rope_path),
            "parent_kv": _trace_parent_kv(trace_path),
            "trusted_tip": state_tip_commitment(envelope, envelope_payload),
        })
    return {
        "schema": PARENT_SCHEMA,
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "tensor_map_sha256": TENSOR_MAP_SHA256,
        },
        "build_manifest_sha256": build_sha256,
        "layers": layers,
    }


def validate_parent_document(document: Mapping[str, Any], expected_sha256: str) -> None:
    require(document.get("schema") == PARENT_SCHEMA, "parent schema mismatch")
    require(
        document.get("model_binding") == {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "tensor_map_sha256": TENSOR_MAP_SHA256,
        },
        "parent checkpoint/vector binding mismatch",
    )
    layers = document.get("layers")
    require(isinstance(layers, list) and len(layers) == LAYER_COUNT, "parent layer count mismatch")
    require([layer.get("layer_index") for layer in layers] == list(range(LAYER_COUNT)), "parent layer order mismatch")
    state_hashes = [layer.get("state", {}).get("sha256") for layer in layers]
    require(len(set(state_hashes)) == LAYER_COUNT, "duplicated parent state")
    for layer_index, layer in enumerate(layers):
        parent_kv = layer.get("parent_kv")
        require(
            isinstance(parent_kv, Mapping)
            and set(parent_kv) == {"k_sha256", "v_sha256", "elements_each", "format"}
            and all(
                isinstance(parent_kv.get(key), str) and len(parent_kv[key]) == 64
                for key in ("k_sha256", "v_sha256")
            )
            and parent_kv.get("elements_each") == KEY_VALUE_HEADS * HEAD_DIM
            and parent_kv.get("format") == "FP16",
            f"layer {layer_index} parent K/V binding mismatch",
        )
    require(sha256(canonical_json(document)) == expected_sha256, "stale parent set digest")


def _copy_verified(source: Path, destination: Path, expected: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    require(hash_file(destination) == expected, f"sealed copy hash mismatch: {destination}")
    destination.chmod(0o444)


def seal_parent_snapshot(
    source_root: Path,
    destination: Path,
    parent_document: Mapping[str, Any],
) -> None:
    require(not destination.exists(), "sealed parent snapshot already exists")
    destination.mkdir(parents=True)
    for record in parent_document["layers"]:
        layer = record["layer_index"]
        state_source = source_root / f"states/layer{layer:02d}/position001"
        transaction_source = source_root / f"transactions/position000/layer{layer:02d}"
        paths = {
            "state": (state_source / "state", destination / f"states/layer{layer:02d}/position001/state"),
            "envelope": (state_source / "envelope.json", destination / f"states/layer{layer:02d}/position001/envelope.json"),
            "transaction": (transaction_source / "transaction.json", destination / f"transactions/position000/layer{layer:02d}/transaction.json"),
            "inputs": (transaction_source / "inputs.hex", destination / f"transactions/position000/layer{layer:02d}/inputs.hex"),
            "final": (transaction_source / "raw/final.hex", destination / f"transactions/position000/layer{layer:02d}/raw/final.hex"),
            "trace": (transaction_source / "raw/trace.hex.gz", destination / f"transactions/position000/layer{layer:02d}/raw/trace.hex.gz"),
            "terminal": (transaction_source / "raw/terminal.txt", destination / f"transactions/position000/layer{layer:02d}/raw/terminal.txt"),
            "rope": (transaction_source / "rope_coefficients.hex", destination / f"transactions/position000/layer{layer:02d}/rope_coefficients.hex"),
        }
        for key, (source, target) in paths.items():
            _copy_verified(source, target, record[key])
    manifest = destination / "manifest.json"
    write_json(manifest, dict(parent_document))
    manifest.chmod(0o444)
    for directory in sorted((path for path in destination.rglob("*") if path.is_dir()), reverse=True):
        directory.chmod(0o555)
    destination.chmod(0o555)


def load_feedback(feedback_dir: Path, contract: Mapping[str, Any]) -> np.ndarray:
    bindings = contract["sealed_feedback"]
    files = {
        "raw": feedback_dir / "raw.txt",
        "terminal": feedback_dir / "terminal.txt",
        "comparison": feedback_dir / "comparison.json",
        "oracle": feedback_dir / "official_vectors/oracle.json",
        "embedding": feedback_dir / "official_vectors/oracle_embedding.hex",
    }
    for key, path in files.items():
        require(path.is_file() and hash_file(path) == bindings[key], f"sealed feedback {key} mismatch")
    comparison = load_json(files["comparison"], "feedback comparison")
    oracle = load_json(files["oracle"], "feedback oracle")
    raw = parse_raw(files["raw"], HIDDEN_SIZE)
    embedding = np.asarray(
        [int(line, 16) for line in files["embedding"].read_text(encoding="ascii").splitlines()],
        dtype="<u2",
    )
    require(
        comparison.get("pass") is True
        and comparison.get("rtl_selected_token_id") == SELECTED_TOKEN
        and comparison.get("next_position") == POSITION
        and oracle.get("selected", {}).get("token_id") == SELECTED_TOKEN
        and oracle.get("model", {}).get("checkpoint_sha256") == CHECKPOINT_SHA256
        and raw["selected"]["token_id"] == SELECTED_TOKEN
        and raw["feedback"] == embedding.tolist()
        and len(embedding) == HIDDEN_SIZE,
        "sealed token-2114 feedback mismatch",
    )
    return embedding


def _negative_checks(document: Mapping[str, Any], expected_sha256: str) -> None:
    variants = []
    missing = copy.deepcopy(document); missing["layers"].pop(); variants.append(missing)
    reordered = copy.deepcopy(document); reordered["layers"][0], reordered["layers"][1] = reordered["layers"][1], reordered["layers"][0]; variants.append(reordered)
    duplicated = copy.deepcopy(document); duplicated["layers"][1]["state"] = duplicated["layers"][0]["state"]; variants.append(duplicated)
    stale = copy.deepcopy(document); stale["layers"][2]["state"]["sha256"] = "0" * 64; variants.append(stale)
    mismatch = copy.deepcopy(document); mismatch["model_binding"]["checkpoint_sha256"] = "0" * 64; variants.append(mismatch)
    for variant in variants:
        try:
            validate_parent_document(variant, expected_sha256)
        except TraversalError:
            continue
        raise TraversalError("negative parent validation unexpectedly passed")


def preflight(
    repository_root: Path,
    source_root: Path,
    parent_compiled_dir: Path,
    compiled_dir: Path,
    checkpoint_path: Path,
    tensor_map_path: Path,
    feedback_dir: Path,
    contract_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(not output_dir.exists(), "semantic K/V preflight directory already exists")
    require(hash_file(checkpoint_path)["sha256"] == CHECKPOINT_SHA256, "checkpoint mismatch")
    require(hash_file(tensor_map_path)["sha256"] == TENSOR_MAP_SHA256, "tensor map mismatch")
    _authenticate_tensor_map(tensor_map_path)
    contract = load_contract(contract_path)
    parent_document = collect_parent_document(
        repository_root, source_root, parent_compiled_dir,
        contract["parent_import"]["build_manifest_sha256"],
    )
    parent_set_sha256 = contract["parent_import"]["parent_set_sha256"]
    validate_parent_document(parent_document, parent_set_sha256)
    _negative_checks(parent_document, parent_set_sha256)
    load_feedback(feedback_dir, contract)
    _, first_voice_contract = contract_binding(repository_root)
    build_manifest, build_sha256 = authenticate_build(
        repository_root, compiled_dir, first_voice_contract
    )
    require(
        build_sha256 == contract["execution_build"]["build_manifest_sha256"],
        "corrected execution build mismatch",
    )
    binary_hashes = compiled_binary_hashes(build_manifest)

    output_dir.mkdir(parents=True)
    sealed = output_dir / "sealed_parents"
    seal_parent_snapshot(source_root, sealed, parent_document)
    preload_dir = output_dir / "semantic_kv"
    layers = []
    for layer_index, parent in enumerate(parent_document["layers"]):
        trace_path = sealed / f"transactions/position000/layer{layer_index:02d}/raw/trace.hex.gz"
        manifest_path, payload_path = write_semantic_kv_preload(
            trace_path,
            preload_dir,
            layer_index=layer_index,
            parent=parent,
            parent_document=parent_document,
            parent_set_sha256=parent_set_sha256,
        )
        readback_path = preload_dir / f"layer{layer_index:02d}.readback.hex"
        log_path = preload_dir / f"layer{layer_index:02d}.preflight.log"
        command = [
            str(compiled_dir / f"layer{layer_index}/bin/Vace3_decoder_layer0_token_engine"),
            "--layer-index", str(layer_index),
            "--semantic-kv-preload", str(payload_path),
            "--semantic-kv-readback", str(readback_path),
            "--semantic-kv-preload-only",
        ]
        with log_path.open("wb") as log:
            completed = subprocess.run(
                command, cwd=repository_root, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        require(completed.returncode == 0, f"layer {layer_index} semantic K/V preflight failed")
        require(
            readback_path.read_bytes() == payload_path.read_bytes()
            and parse_semantic_kv_payload(readback_path, layer_index) == parent["parent_kv"],
            f"layer {layer_index} semantic K/V readback mismatch",
        )
        layers.append({
            "layer_index": layer_index,
            "binary_sha256": binary_hashes[layer_index],
            "manifest": hash_file(manifest_path),
            "payload": hash_file(payload_path),
            "readback": hash_file(readback_path),
            "log": hash_file(log_path),
            "exact_readback": True,
        })
    result = {
        "schema": "ace3-semantic-kv-preflight-v1",
        "model_binding": parent_document["model_binding"],
        "parent_set_sha256": parent_set_sha256,
        "execution_build_manifest_sha256": build_sha256,
        "selected_token": SELECTED_TOKEN,
        "source_position": 0,
        "execution_position": POSITION,
        "layers": layers,
        "official_traversal_consumed": False,
    }
    write_json(output_dir / "result.json", result)
    return result


def execute(
    repository_root: Path,
    source_root: Path,
    parent_compiled_dir: Path,
    compiled_dir: Path,
    checkpoint_path: Path,
    tensor_map_path: Path,
    feedback_dir: Path,
    contract_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(not output_dir.exists(), "position-1 output directory already exists")
    require(hash_file(checkpoint_path)["sha256"] == CHECKPOINT_SHA256, "checkpoint mismatch")
    require(hash_file(tensor_map_path)["sha256"] == TENSOR_MAP_SHA256, "tensor map mismatch")
    contract = load_contract(contract_path)
    parent_document = collect_parent_document(
        repository_root,
        source_root,
        parent_compiled_dir,
        contract["parent_import"]["build_manifest_sha256"],
    )
    expected_parent_sha = contract["parent_import"]["parent_set_sha256"]
    validate_parent_document(parent_document, expected_parent_sha)
    _negative_checks(parent_document, expected_parent_sha)
    for record in parent_document["layers"]:
        layer = record["layer_index"]
        validate_state_envelope(
            source_root / f"states/layer{layer:02d}/position001/envelope.json",
            source_root / f"states/layer{layer:02d}/position001/state",
            states_dir=source_root / "states",
            runtime_dir=source_root / "transactions",
            layer_index=layer,
            next_position=POSITION,
            build_manifest_sha256=parent_document["build_manifest_sha256"],
            binary_sha256=record["binary_sha256"],
            expected_tip=record["trusted_tip"],
        )
    embedding_bits = load_feedback(feedback_dir, contract)

    output_dir.mkdir(parents=True)
    sealed = output_dir / "sealed_parents"
    seal_parent_snapshot(source_root, sealed, parent_document)
    execution = output_dir / "execution"
    preload_dir = output_dir / "semantic_kv"

    tensor_map = _authenticate_tensor_map(tensor_map_path)
    _, first_voice_contract = contract_binding(repository_root)
    build_manifest, build_sha = authenticate_build(repository_root, compiled_dir, first_voice_contract)
    require(
        build_sha == contract["execution_build"]["build_manifest_sha256"],
        "corrected execution build mismatch",
    )
    binary_hashes = compiled_binary_hashes(build_manifest)
    trusted_tips: dict[int, dict[str, Any]] = {}

    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    embeddings, _, reference_embeddings, reference_states = _load_model(checkpoint_path)
    require(
        np.array_equal(_f16_to_bits(embeddings[SELECTED_TOKEN]), embedding_bits),
        "Host-substituted token embedding",
    )
    hidden_bits = embedding_bits
    contract_reference_hidden = torch.from_numpy(
        _bits_to_f16(embedding_bits).astype(np.float64)
    )
    continuous_reference_hidden = reference_embeddings[SELECTED_TOKEN].clone()
    vector_workspace = output_dir / "vector_workspace"
    tensor_manifests = output_dir / "tensor_manifests"
    tensor_manifests.mkdir()
    layers = []
    for layer_index, reference_state in enumerate(reference_states):
        input_bits = hidden_bits.copy()
        parent = parent_document["layers"][layer_index]
        parent_kv, reference_k, reference_v = _load_parent_kv(
            sealed / f"transactions/position000/layer{layer_index:02d}/raw/trace.hex.gz",
            parent["parent_kv"],
            layer_index,
        )
        require(reference_state.layer_id == layer_index, "reference layer order mismatch")
        require(
            reference_state.reference_k.shape[0] == 0
            and reference_state.reference_v.shape[0] == 0,
            f"layer {layer_index} reference cache was not empty before import",
        )
        preload_manifest, preload_payload = write_semantic_kv_preload(
            sealed / f"transactions/position000/layer{layer_index:02d}/raw/trace.hex.gz",
            preload_dir,
            layer_index=layer_index,
            parent=parent,
            parent_document=parent_document,
            parent_set_sha256=expected_parent_sha,
        )
        preload_readback = preload_dir / f"layer{layer_index:02d}.readback.hex"
        tensor_manifest = _serialize_layer_tensors(
            checkpoint_path, tensor_map, layer_index, vector_workspace, tensor_manifests
        )
        try:
            hidden_bits, transaction = _run_transaction(
                repository_root=repository_root,
                compiled_dir=compiled_dir,
                vector_dir=vector_workspace,
                runtime_dir=execution / "transactions",
                states_dir=execution / "states",
                layer_index=layer_index,
                position=POSITION,
                hidden_bits=hidden_bits,
                build_manifest_sha256=build_sha,
                binary_sha256=binary_hashes[layer_index],
                trusted_tips=trusted_tips,
                semantic_kv_preload_path=preload_payload,
                semantic_kv_readback_path=preload_readback,
            )
        finally:
            shutil.rmtree(vector_workspace, ignore_errors=True)
        (
            contract_reference_hidden,
            continuous_reference_hidden,
            local_reference_hidden,
            contract_comparison,
            continuous_comparison,
            local_comparison,
        ) = _layer_reference_comparisons(
            reference_state,
            contract_reference_hidden,
            continuous_reference_hidden,
            input_bits,
            hidden_bits,
            reference_k,
            reference_v,
            POSITION,
        )
        require(
            preload_readback.read_bytes() == preload_payload.read_bytes()
            and parse_semantic_kv_payload(preload_readback, layer_index) == parent["parent_kv"],
            f"layer {layer_index} semantic K/V readback mismatch",
        )
        require(
            contract_comparison["max_abs_error"]
            <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
            f"layer {layer_index} independent oracle mismatch",
        )
        layers.append({
            "layer_index": layer_index,
            "semantic_parent_state_binding": parent["state"],
            "semantic_parent_envelope_binding": parent["envelope"],
            "semantic_parent_kv": parent["parent_kv"],
            "semantic_parent_binary_sha256": parent["binary_sha256"],
            "semantic_kv_manifest": hash_file(preload_manifest),
            "semantic_kv_payload": hash_file(preload_payload),
            "semantic_kv_readback": hash_file(preload_readback),
            "binary_sha256": binary_hashes[layer_index],
            "tensor_manifest": tensor_manifest,
            "transaction": transaction,
            "independent_reference": {
                **contract_comparison,
                "seed": "selected token embedding; prior RTL hidden is never consumed",
                "inter_layer_boundary": "independent binary16 round after every reference layer",
                "inherited_parent_kv": parent_kv,
                "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
                "within_tolerance": True,
                "gating": True,
            },
            "continuous_float64_reference": {
                **continuous_comparison,
                "seed": "float64 reference propagated from token embedding",
                "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
                "within_tolerance": (
                    continuous_comparison["max_abs_error"]
                    <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE
                ),
                "gating": False,
            },
            "local_reference": {
                **local_comparison,
                "seed": "actual layer input from prior RTL FP16 output",
                "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
                "within_tolerance": (
                    local_comparison["max_abs_error"]
                    <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE
                ),
                "gating": False,
            },
        })
    result = {
        "schema": SCHEMA,
        "selected_token": SELECTED_TOKEN,
        "position": POSITION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "tensor_map_sha256": TENSOR_MAP_SHA256,
        "build_manifest_sha256": build_sha,
        "parent_build_manifest_sha256": parent_document["build_manifest_sha256"],
        "parent_set_sha256": expected_parent_sha,
        "sealed_snapshot": hash_file(sealed / "manifest.json"),
        "layers": layers,
        "terminal_hidden_sha256": sha256(np.asarray(hidden_bits, dtype="<u2").tobytes()),
        "natural_terminal_layers": sum(int(layer["transaction"]["natural_terminal"]) for layer in layers),
        "hard_gate": "embedding-seeded contract-precision cumulative reference only",
        "claim_boundary": "token 2114 at position 1 only; no lm_head rerun, dialogue, synthesis, PPA, FPGA, or latency claim",
    }
    write_json(output_dir / "result.json", result)
    return result


def verify_result(path: Path, contract_path: Path) -> dict[str, Any]:
    result = load_json(path, "position-1 result")
    contract = load_contract(contract_path)
    require(result.get("schema") == SCHEMA, "result schema mismatch")
    require(result.get("selected_token") == SELECTED_TOKEN and result.get("position") == POSITION, "result token/position mismatch")
    require(result.get("checkpoint_sha256") == CHECKPOINT_SHA256 and result.get("tensor_map_sha256") == TENSOR_MAP_SHA256, "result checkpoint/vector mismatch")
    require(result.get("build_manifest_sha256") == contract["execution_build"]["build_manifest_sha256"], "result execution build mismatch")
    require(result.get("parent_build_manifest_sha256") == contract["parent_import"]["build_manifest_sha256"], "result parent build mismatch")
    require(result.get("parent_set_sha256") == contract["parent_import"]["parent_set_sha256"], "result parent binding mismatch")
    layers = result.get("layers")
    require(isinstance(layers, list) and [x.get("layer_index") for x in layers] == list(range(LAYER_COUNT)), "result layer order mismatch")
    require(result.get("natural_terminal_layers") == LAYER_COUNT, "result natural terminal mismatch")
    require(
        result.get("hard_gate")
        == "embedding-seeded contract-precision cumulative reference only",
        "result hard gate mismatch",
    )
    require(
        all(
            x.get("transaction", {}).get("natural_terminal") is True
            and x.get("transaction", {}).get("final_records") == HIDDEN_SIZE
            and x.get("transaction", {}).get("done_records") == 1
            and x.get("transaction", {}).get("semantic_kv_readback") == "exact"
            and x.get("semantic_kv_payload") == x.get("semantic_kv_readback")
            and x.get("independent_reference", {}).get("seed")
            == "selected token embedding; prior RTL hidden is never consumed"
            and x.get("independent_reference", {}).get("inter_layer_boundary")
            == "independent binary16 round after every reference layer"
            and x.get("independent_reference", {}).get("absolute_tolerance")
            == TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE
            and type(x.get("independent_reference", {}).get("max_abs_error"))
            in (int, float)
            and x["independent_reference"]["max_abs_error"]
            <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE
            and x.get("independent_reference", {}).get("within_tolerance") is True
            and x.get("independent_reference", {}).get("gating") is True
            and x.get("continuous_float64_reference", {}).get("gating") is False
            and x.get("local_reference", {}).get("gating") is False
            for x in layers
        ),
        "result oracle mismatch",
    )
    return result


def recompute_preserved(
    preserved_output_dir: Path,
    checkpoint_path: Path,
    feedback_dir: Path,
    contract_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    require(not report_path.exists(), "preserved comparison report already exists")
    require(hash_file(checkpoint_path)["sha256"] == CHECKPOINT_SHA256, "checkpoint mismatch")
    contract = load_contract(contract_path)
    embedding_bits = load_feedback(feedback_dir, contract)
    failure_path = preserved_output_dir / "failure_terminal_manifest.json"
    failure = load_json(failure_path, "preserved repair7 failure terminal")
    require(
        failure.get("schema") == "ace3-position1-terminal-failure-v1"
        and failure.get("attempt", {}).get("replayed") is False
        and failure.get("evidence", {}).get("all_24_natural_terminal") is True
        and failure.get("evidence", {}).get("all_24_semantic_kv_readbacks_exact") is True,
        "preserved repair7 terminal binding mismatch",
    )
    parent_document = load_json(
        preserved_output_dir / "sealed_parents/manifest.json",
        "preserved sealed parent manifest",
    )
    validate_parent_document(
        parent_document, contract["parent_import"]["parent_set_sha256"]
    )
    embeddings, _, reference_embeddings, reference_states = _load_model(checkpoint_path)
    require(
        np.array_equal(_f16_to_bits(embeddings[SELECTED_TOKEN]), embedding_bits),
        "Host-substituted token embedding",
    )
    contract_hidden = torch.from_numpy(
        _bits_to_f16(embedding_bits).astype(np.float64)
    )
    continuous_hidden = reference_embeddings[SELECTED_TOKEN].clone()
    previous_output_bits: np.ndarray | None = None
    layers = []
    for layer_index, reference_state in enumerate(reference_states):
        transaction_dir = (
            preserved_output_dir
            / f"execution/transactions/position001/layer{layer_index:02d}"
        )
        transaction = load_json(
            transaction_dir / "transaction.json",
            f"preserved layer {layer_index} transaction",
        )
        require(
            transaction.get("layer_index") == layer_index
            and transaction.get("position") == POSITION
            and transaction.get("natural_terminal") is True
            and transaction.get("final_records") == HIDDEN_SIZE
            and transaction.get("done_records") == 1,
            f"preserved layer {layer_index} natural terminal mismatch",
        )
        input_path = transaction_dir / "inputs.hex"
        output_path = transaction_dir / "raw/final.hex"
        input_bits = _parse_hidden_words(input_path, f"layer {layer_index} input")
        output_bits = _parse_hidden_words(output_path, f"layer {layer_index} output")
        require(
            np.array_equal(
                input_bits,
                embedding_bits if previous_output_bits is None else previous_output_bits,
            ),
            f"preserved layer {layer_index} activation chain mismatch",
        )
        parent = parent_document["layers"][layer_index]
        parent_kv, reference_k, reference_v = _load_parent_kv(
            preserved_output_dir
            / f"sealed_parents/transactions/position000/layer{layer_index:02d}/raw/trace.hex.gz",
            parent["parent_kv"],
            layer_index,
        )
        preload = preserved_output_dir / f"semantic_kv/layer{layer_index:02d}.hex"
        readback = preserved_output_dir / f"semantic_kv/layer{layer_index:02d}.readback.hex"
        require(
            preload.read_bytes() == readback.read_bytes()
            and parse_semantic_kv_payload(readback, layer_index) == parent["parent_kv"],
            f"preserved layer {layer_index} semantic K/V readback mismatch",
        )
        (
            contract_hidden,
            continuous_hidden,
            _,
            contract_comparison,
            continuous_comparison,
            local_comparison,
        ) = _layer_reference_comparisons(
            reference_state,
            contract_hidden,
            continuous_hidden,
            input_bits,
            output_bits,
            reference_k,
            reference_v,
            POSITION,
        )
        layers.append({
            "layer_index": layer_index,
            "input": hash_file(input_path),
            "output": hash_file(output_path),
            "parent_kv": parent_kv,
            "independent_reference": {
                **contract_comparison,
                "seed": "selected token embedding; prior RTL hidden is never consumed",
                "inter_layer_boundary": "independent binary16 round after every reference layer",
                "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
                "within_tolerance": contract_comparison["max_abs_error"]
                <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
                "gating": True,
            },
            "continuous_float64_reference": {
                **continuous_comparison,
                "gating": False,
            },
            "local_reference": {**local_comparison, "gating": False},
        })
        previous_output_bits = output_bits
    accepted = all(
        layer["independent_reference"]["within_tolerance"] for layer in layers
    )
    report = {
        "schema": "ace3-position1-preserved-contract-precision-v1",
        "preserved_repair7": hash_file(failure_path),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "parent_set_sha256": contract["parent_import"]["parent_set_sha256"],
        "hard_gate": "embedding-seeded contract-precision cumulative reference only",
        "layers": layers,
        "layer23_max_abs_error": layers[-1]["independent_reference"]["max_abs_error"],
        "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
        "accepted": accepted,
        "repair7_replayed": False,
    }
    write_json(report_path, report)
    require(accepted, "preserved repair7 contract-precision oracle mismatch")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("repository-root", "source-root", "parent-compiled-dir", "compiled-dir", "checkpoint", "tensor-map", "feedback-dir", "contract", "output-dir"):
        run.add_argument(f"--{name}", type=Path, required=True)
    check = sub.add_parser("preflight")
    for name in ("repository-root", "source-root", "parent-compiled-dir", "compiled-dir", "checkpoint", "tensor-map", "feedback-dir", "contract", "output-dir"):
        check.add_argument(f"--{name}", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--contract", type=Path, required=True)
    preserved = sub.add_parser("recompute-preserved")
    preserved.add_argument("--preserved-output-dir", type=Path, required=True)
    preserved.add_argument("--checkpoint", type=Path, required=True)
    preserved.add_argument("--feedback-dir", type=Path, required=True)
    preserved.add_argument("--contract", type=Path, required=True)
    preserved.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(
            args.repository_root.resolve(strict=True),
            args.source_root.resolve(strict=True),
            args.parent_compiled_dir.resolve(strict=True),
            args.compiled_dir.resolve(strict=True),
            args.checkpoint.resolve(strict=True),
            args.tensor_map.resolve(strict=True),
            args.feedback_dir.resolve(strict=True),
            args.contract.resolve(strict=True),
            args.output_dir.resolve(),
        )
        print(f"POSITION1_SEMANTIC_KV_PREFLIGHT_PASS layers={len(result['layers'])} readback=exact traversal_consumed=0")
    elif args.command == "run":
        result = execute(
            args.repository_root.resolve(strict=True),
            args.source_root.resolve(strict=True),
            args.parent_compiled_dir.resolve(strict=True),
            args.compiled_dir.resolve(strict=True),
            args.checkpoint.resolve(strict=True),
            args.tensor_map.resolve(strict=True),
            args.feedback_dir.resolve(strict=True),
            args.contract.resolve(strict=True),
            args.output_dir.resolve(),
        )
        print(f"POSITION1_MODEL24_CAUSAL_TRAVERSAL_PASS token=2114 position=1 layers={len(result['layers'])} parent_set={result['parent_set_sha256']} terminal={result['terminal_hidden_sha256']}")
    elif args.command == "verify":
        result = verify_result(args.result.resolve(strict=True), args.contract.resolve(strict=True))
        print(f"POSITION1_MODEL24_CAUSAL_VERIFY_PASS layers={len(result['layers'])} natural_terminal=24 oracle=independent")
    else:
        result = recompute_preserved(
            args.preserved_output_dir.resolve(strict=True),
            args.checkpoint.resolve(strict=True),
            args.feedback_dir.resolve(strict=True),
            args.contract.resolve(strict=True),
            args.report.resolve(),
        )
        print(
            "POSITION1_PRESERVED_CONTRACT_PRECISION_PASS "
            f"layers={len(result['layers'])} "
            f"layer23_max_abs_error={result['layer23_max_abs_error']}"
        )


if __name__ == "__main__":
    main()
