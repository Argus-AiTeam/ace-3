#!/usr/bin/env python3
"""Validate next-position traversal inputs without creating launch authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping


HIDDEN_ELEMENTS = 896
MODEL_LAYERS = 24
TRANSACTION_COUNT = 26
ROPE_PAIRS = 32
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
READINESS_KIND_PATTERN = re.compile(
    r"^ace3_cursor[0-9]+_next_position_runtime_readiness$"
)


class PreflightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    require(path.is_absolute(), f"artifact path is not absolute: {path}")
    require(path.is_file(), f"artifact is missing: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as artifact:
        while chunk := artifact.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "path": str(path), "sha256": digest.hexdigest()}


def record_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bytes": record.get("bytes"),
        "path": record.get("path"),
        "sha256": record.get("sha256"),
    }


def authenticate_record(record: Any, label: str) -> dict[str, Any]:
    require(isinstance(record, dict), f"{label} record is malformed")
    require(
        type(record.get("bytes")) is int and record["bytes"] >= 0,
        f"{label} byte count is malformed",
    )
    require(
        isinstance(record.get("path"), str),
        f"{label} path is malformed",
    )
    require(
        isinstance(record.get("sha256"), str)
        and SHA256_PATTERN.fullmatch(record["sha256"]) is not None,
        f"{label} SHA256 is malformed",
    )
    actual = file_record(Path(record["path"]))
    require(record_fields(record) == actual, f"{label} content binding mismatch")
    return actual


def same_content_record(left: Any, right: Any, label: str) -> None:
    require(
        isinstance(left, dict)
        and isinstance(right, dict)
        and left.get("bytes") == right.get("bytes")
        and left.get("sha256") == right.get("sha256"),
        f"{label} content records differ",
    )


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"{label} is not readable canonical JSON: {path}") from error
    require(isinstance(document, dict), f"{label} root is not an object")
    return document


def load_json_record(record: Any, label: str) -> dict[str, Any]:
    authenticated = authenticate_record(record, label)
    return load_json(Path(authenticated["path"]), label)


def semantic_embedding_sha256(path: Path, elements: int) -> str:
    try:
        rows = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PreflightError(f"selected-token feedback is not ASCII: {path}") from error
    require(
        len(rows) == elements,
        f"selected-token feedback row count differs: {len(rows)}",
    )
    payload = bytearray()
    for index, row in enumerate(rows):
        require(
            len(row) == 4 and all(character in "0123456789abcdef" for character in row),
            f"selected-token feedback row {index} is malformed",
        )
        payload.extend(int(row, 16).to_bytes(2, "little"))
    return sha256_bytes(bytes(payload))


def semantic_hidden_sha256(path: Path, elements: int = HIDDEN_ELEMENTS) -> str:
    try:
        rows = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PreflightError(f"hidden vector is not ASCII: {path}") from error
    require(len(rows) == elements, f"hidden vector row count differs: {len(rows)}")
    payload = bytearray()
    for index, row in enumerate(rows):
        require(
            len(row) == 10
            and row[:2] == "00"
            and all(character in "0123456789abcdef" for character in row)
            and int(row[2:6], 16) == index,
            f"hidden vector row {index} ordering or encoding differs",
        )
        payload.extend(int(row[6:10], 16).to_bytes(2, "little"))
    return sha256_bytes(bytes(payload))


def validate_rope_rows(path: Path, fixture_position: int) -> int:
    try:
        rows = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PreflightError(f"RoPE coefficients are not ASCII: {path}") from error
    expected = [
        (position, pair)
        for position in range(fixture_position + 1)
        for pair in range(ROPE_PAIRS)
    ]
    require(len(rows) == len(expected), "RoPE coefficient row count differs")
    for index, (row, indices) in enumerate(zip(rows, expected, strict=True)):
        require(
            len(row) == 14
            and all(character in "0123456789abcdef" for character in row),
            f"RoPE coefficient row {index} is malformed",
        )
        require(
            (int(row[:4], 16), int(row[4:6], 16)) == indices,
            f"RoPE coefficient row {index} index differs",
        )
    return len(rows)


def required_tensor_names(layer_index: int) -> set[str]:
    prefix = f"model.layers.{layer_index}"
    return {
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        *{
            f"{prefix}.mlp.{projection}.{field}"
            for projection in ("gate_proj", "up_proj", "down_proj")
            for field in ("qweight", "qzeros", "scales")
        },
        *{
            f"{prefix}.self_attn.{projection}.{field}"
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            for field in ("qweight", "qzeros", "scales")
        },
        *{
            f"{prefix}.self_attn.{projection}.bias"
            for projection in ("q_proj", "k_proj", "v_proj")
        },
    }


def validate_fixture(
    fixture_record: Any,
    *,
    layer_index: int,
    transaction_position: int,
) -> tuple[dict[str, Any], int]:
    fixture = load_json_record(fixture_record, "next-layer fixture manifest")
    tensors = fixture.get("tensors")
    fixture_position = transaction_position - 1
    required_names = required_tensor_names(layer_index)
    require(
        fixture.get("schema_version") == 1
        and fixture.get("kind") == "ace3_position2_live_transaction_vectors"
        and fixture.get("layer_index") == layer_index
        and fixture.get("position") == fixture_position
        and isinstance(fixture.get("input_activation_sha256"), str)
        and SHA256_PATTERN.fullmatch(fixture["input_activation_sha256"]) is not None
        and isinstance(tensors, list)
        and len(tensors) == len(required_names),
        "next-layer fixture identity, position, or tensor count differs",
    )
    input_record = authenticate_record(fixture.get("input"), "fixture input")
    require(
        semantic_hidden_sha256(Path(input_record["path"]))
        == fixture["input_activation_sha256"],
        "fixture input semantic SHA256 differs",
    )
    rope_record = authenticate_record(
        fixture.get("rope_coefficients"),
        "fixture RoPE coefficients",
    )
    rope_rows = validate_rope_rows(Path(rope_record["path"]), fixture_position)

    observed: set[str] = set()
    for index, tensor in enumerate(tensors):
        require(isinstance(tensor, dict), f"fixture tensor {index} is malformed")
        checkpoint = tensor.get("checkpoint_tensor")
        serialized = tensor.get("serialized")
        require(
            isinstance(checkpoint, dict)
            and set(checkpoint) == {"bytes", "dtype", "name", "sha256", "shape"}
            and isinstance(checkpoint.get("name"), str)
            and checkpoint["name"] in required_names
            and checkpoint["name"] not in observed
            and checkpoint.get("dtype") in {"F16", "I32"}
            and isinstance(checkpoint.get("shape"), list)
            and checkpoint["shape"]
            and all(
                type(dimension) is int and dimension > 0
                for dimension in checkpoint["shape"]
            )
            and isinstance(checkpoint.get("sha256"), str)
            and SHA256_PATTERN.fullmatch(checkpoint["sha256"]) is not None,
            f"fixture tensor {index} checkpoint metadata differs",
        )
        elements = 1
        for dimension in checkpoint["shape"]:
            elements *= dimension
        unit_bytes = 4 if checkpoint["dtype"] == "I32" else 2
        require(
            checkpoint.get("bytes") == elements * unit_bytes,
            f"fixture tensor {index} checkpoint byte count differs",
        )
        expected_dtype = (
            "I32"
            if checkpoint["name"].endswith((".qweight", ".qzeros"))
            else "F16"
        )
        require(
            checkpoint["dtype"] == expected_dtype,
            f"fixture tensor {checkpoint['name']} dtype differs",
        )
        authenticate_record(serialized, f"fixture tensor {checkpoint['name']}")
        observed.add(checkpoint["name"])
    require(observed == required_names, "next-layer fixture tensor names differ")
    return fixture, rope_rows


def validate_selected_token_feedback(
    document: Mapping[str, Any],
    *,
    generation_manifest: Mapping[str, Any],
    ledger: Mapping[str, Any],
    transaction_position: int,
) -> tuple[int, str]:
    cursor = document["cursor"]
    files = generation_manifest.get("files")
    completed = ledger.get("completed_receipts")
    require(
        isinstance(files, dict)
        and isinstance(completed, list)
        and len(completed) >= 2,
        "selected-token parent records are missing",
    )
    receipt0_record = cursor.get("checkpoint000") or files.get(
        "checkpoints/transaction-000.json"
    )
    receipt1_record = document.get("embedding_feedback", {}).get(
        "layer0_consumed_receipt"
    )
    require(
        receipt0_record == files.get("checkpoints/transaction-000.json")
        and receipt1_record == files.get("checkpoints/transaction-001.json")
        and completed[0] == receipt0_record
        and completed[1] == receipt1_record,
        "selected-token receipt lineage differs from generation state",
    )
    receipt0 = load_json_record(receipt0_record, "selected-token receipt")
    receipt1 = load_json_record(receipt1_record, "layer-0 feedback receipt")
    selected = document.get("selected_token")
    feedback = document.get("embedding_feedback")
    require(
        isinstance(selected, dict) and isinstance(feedback, dict),
        "selected-token feedback sections are missing",
    )
    vector_record = authenticate_record(feedback.get("vector"), "feedback vector")
    semantics = feedback.get("semantics")
    require(
        isinstance(semantics, dict)
        and semantics.get("dtype") == "FP16"
        and semantics.get("elements") == HIDDEN_ELEMENTS
        and type(semantics.get("token_id")) is int
        and type(semantics.get("selected_logit_f16_bits")) is int
        and isinstance(semantics.get("semantic_sha256"), str)
        and SHA256_PATTERN.fullmatch(semantics["semantic_sha256"]) is not None,
        "selected-token feedback semantics differ",
    )
    require(
        semantic_embedding_sha256(Path(vector_record["path"]), HIDDEN_ELEMENTS)
        == semantics["semantic_sha256"],
        "selected-token feedback semantic SHA256 differs",
    )
    receipt0_outputs = receipt0.get("outputs", {}).get("selected_token_embedding")
    receipt0_semantics = receipt0.get("output_semantics", {}).get(
        "selected_token_embedding"
    )
    require(
        receipt0.get("status") == "COMPLETE"
        and receipt0.get("transaction_index") == 0
        and receipt0_outputs == feedback["vector"]
        and receipt0_semantics == semantics
        and receipt0.get("result", {}).get("selected_token_id")
        == semantics["token_id"]
        and receipt0.get("result", {}).get("selected_logit_f16_bits")
        == semantics["selected_logit_f16_bits"],
        "selected-token receipt output or semantics differ",
    )
    require(
        selected.get("lm_head_receipt") == receipt0_record
        and selected.get("token_id") == semantics["token_id"]
        and selected.get("selected_logit_f16_bits")
        == semantics["selected_logit_f16_bits"],
        "selected-token readiness summary differs from its receipt",
    )
    predecessor = receipt1.get("authenticated_inputs", {}).get("predecessor")
    state_semantics = receipt1.get("output_semantics", {}).get("state")
    require(
        receipt1.get("status") == "COMPLETE"
        and receipt1.get("transaction_index") == 1
        and isinstance(state_semantics, dict)
        and state_semantics.get("layer_index") == 0
        and state_semantics.get("position") == transaction_position + 1
        and state_semantics.get("parent_position") == transaction_position - 1
        and receipt1.get("authenticated_inputs", {}).get("transaction_position")
        == transaction_position
        and isinstance(predecessor, dict)
        and predecessor.get("state") == "AUTHENTICATED"
        and predecessor.get("source_transaction_index") == 0
        and predecessor.get("embedding") == feedback["vector"]
        and predecessor.get("embedding_semantics") == semantics
        and predecessor.get("selected_token_id") == semantics["token_id"]
        and predecessor.get("selected_logit_f16_bits")
        == semantics["selected_logit_f16_bits"],
        "layer-0 selected-token feedback consumption differs",
    )
    same_content_record(
        predecessor.get("source_receipt"),
        receipt0_record,
        "layer-0 source receipt",
    )
    require(
        semantics.get("checkpoint")
        == receipt0.get("authenticated_inputs", {}).get("checkpoint"),
        "selected-token checkpoint binding differs",
    )
    return semantics["token_id"], semantics["semantic_sha256"]


def validate_parent_receipt(
    receipt_record: Any,
    state_record: Any,
    *,
    parent_transaction_index: int,
    layer_index: int,
    output_state_position: int,
) -> str:
    receipt = load_json_record(receipt_record, "parent checkpoint receipt")
    result = receipt.get("result")
    receipt_layer_index = receipt.get("layer_index")
    if receipt_layer_index is None:
        receipt_layer_index = receipt.get("output_semantics", {}).get(
            "state", {}
        ).get("layer_index")
    require(
        receipt.get("status") == "COMPLETE"
        and receipt.get("transaction_index") == parent_transaction_index
        and receipt_layer_index == layer_index - 1
        and isinstance(result, dict)
        and result.get("exact_integer_oracle_match") is True
        and result.get("natural_rtl_terminal") is True
        and result.get("output_hidden_elements") == HIDDEN_ELEMENTS,
        "parent checkpoint completion semantics differ",
    )
    receipt_state = receipt.get("outputs", {}).get("state") or result.get(
        "output_state"
    )
    require(
        receipt_state == state_record,
        "parent checkpoint hidden/KV state binding differs",
    )
    state = authenticate_record(state_record, "parent checkpoint hidden/KV state")
    require(
        Path(state["path"]).name == f"position{output_state_position:03d}.state",
        "parent checkpoint state position differs",
    )
    if "output_state_position" in result:
        require(
            result["output_state_position"] == output_state_position,
            "parent checkpoint result state position differs",
        )

    hidden_record = receipt.get("outputs", {}).get("hidden")
    reconstructed = False
    if hidden_record is None:
        hidden_record = receipt.get("rtl_reference_agreement", {}).get("rtl_final")
        reconstructed = True
    hidden = authenticate_record(hidden_record, "parent checkpoint hidden output")
    semantic_sha256 = semantic_hidden_sha256(Path(hidden["path"]))
    output_semantics = receipt.get("output_semantics", {}).get("hidden")
    if output_semantics is not None:
        require(
            output_semantics
            == {
                "dtype": "FP16",
                "elements": HIDDEN_ELEMENTS,
                "semantic_sha256": semantic_sha256,
            },
            "parent checkpoint hidden semantics differ",
        )
    if reconstructed:
        agreement = receipt.get("rtl_reference_agreement")
        require(
            isinstance(agreement, dict)
            and agreement.get("compared_elements") == HIDDEN_ELEMENTS
            and agreement.get("mismatches") == 0,
            "reconstructed parent hidden comparison differs",
        )
        oracle_hidden = authenticate_record(
            agreement.get("exact_integer_oracle_final"),
            "parent checkpoint oracle hidden output",
        )
        same_content_record(
            oracle_hidden,
            hidden_record,
            "parent checkpoint RTL/oracle hidden",
        )
    return semantic_sha256


def validate_preflight_document(document: Mapping[str, Any]) -> dict[str, Any]:
    require(
        document.get("schema_version") == 1
        and isinstance(document.get("kind"), str)
        and READINESS_KIND_PATTERN.fullmatch(document["kind"]) is not None
        and document.get("data_bindings_ready") is True
        and document.get("launchable") is False
        and document.get("status") == "NOT_READY"
        and document.get("execution_performed_by_this_check") is False
        and isinstance(document.get("missing_required_bindings"), list)
        and document["missing_required_bindings"],
        "readiness claim boundary does not remain fail closed",
    )
    runtime_identity = document.get("runtime_identity")
    cursor = document.get("cursor")
    require(
        isinstance(runtime_identity, str)
        and runtime_identity
        and isinstance(cursor, dict),
        "runtime identity or cursor is missing",
    )
    generation = cursor.get("generation")
    next_transaction = cursor.get("next_transaction_index")
    parent_transaction = cursor.get("checkpoint")
    require(
        type(generation) is int
        and type(next_transaction) is int
        and type(parent_transaction) is int
        and generation == next_transaction
        and parent_transaction == next_transaction - 1
        and cursor.get("parent_generation") == generation - 1
        and 1 <= next_transaction <= MODEL_LAYERS,
        "cursor, generation, or parent checkpoint bounds differ",
    )
    parent_key = f"checkpoint{parent_transaction:03d}"
    parent_record = cursor.get(parent_key)
    require(isinstance(parent_record, dict), f"{parent_key} binding is missing")

    pointer = load_json_record(
        cursor.get("authoritative_pointer"),
        "authoritative cursor pointer",
    )
    generation_manifest = load_json_record(
        cursor.get("generation_manifest"),
        "generation manifest",
    )
    ledger = load_json_record(cursor.get("ledger"), "generation ledger")
    require(
        pointer.get("status") == "COMMITTED"
        and pointer.get("runtime_identity") == runtime_identity
        and pointer.get("generation") == generation
        and pointer.get("generation_manifest") == cursor["generation_manifest"],
        "authoritative pointer does not select the recorded generation",
    )
    files = generation_manifest.get("files")
    future_absence_key = f"transactions{next_transaction:03d}_025_executed"
    require(
        generation_manifest.get("status") == "PREPARED"
        and generation_manifest.get("runtime_identity") == runtime_identity
        and generation_manifest.get("generation") == generation
        and generation_manifest.get("parent_generation") == generation - 1
        and generation_manifest.get(future_absence_key) is False
        and isinstance(files, dict)
        and files.get("ledger.json") == cursor["ledger"]
        and files.get(
            f"checkpoints/transaction-{parent_transaction:03d}.json"
        )
        == parent_record,
        "generation manifest parentage or future-execution state differs",
    )
    completed = ledger.get("completed_receipts")
    require(
        ledger.get("runtime_identity") == runtime_identity
        and ledger.get("state_generation") == generation
        and ledger.get("completed_transaction_count") == next_transaction
        and ledger.get("next_transaction_index") == next_transaction
        and ledger.get("transaction_count") == TRANSACTION_COUNT
        and isinstance(completed, list)
        and len(completed) == next_transaction
        and completed[parent_transaction] == parent_record,
        "ledger cursor or completed parentage differs",
    )
    invocation_record = ledger.get("invocation")
    invocation = load_json_record(invocation_record, "bound invocation")
    transactions = invocation.get("transactions")
    require(
        invocation.get("runtime_identity") == runtime_identity
        and isinstance(transactions, list)
        and len(transactions) == TRANSACTION_COUNT,
        "bound invocation identity or transaction count differs",
    )
    descriptor = transactions[next_transaction]
    next_inputs = document.get("next_layer_inputs")
    require(
        isinstance(descriptor, dict)
        and isinstance(next_inputs, dict),
        "next transaction descriptor or readiness inputs are missing",
    )
    layer_index = descriptor.get("layer_index")
    transaction_position = descriptor.get("inputs", {}).get(
        "transaction_position"
    )
    output_state_position = descriptor.get("required_outputs", {}).get(
        "state", {}
    ).get("position")
    require(
        descriptor.get("transaction_index") == next_transaction
        and descriptor.get("operation") == "position3-decoder-layer"
        and type(layer_index) is int
        and layer_index == next_transaction - 1
        and 0 <= layer_index < MODEL_LAYERS
        and descriptor.get("inputs", {}).get("predecessor", {}).get(
            "source_transaction_index"
        )
        == parent_transaction
        and type(transaction_position) is int
        and transaction_position >= 1
        and output_state_position == transaction_position + 1
        and descriptor.get("required_outputs", {}).get("state", {}).get(
            "layer_index"
        )
        == layer_index
        and descriptor.get("required_outputs", {}).get("hidden")
        == {
            "dtype": "FP16",
            "elements": HIDDEN_ELEMENTS,
            "semantic_sha256": "required sha256",
        },
        "next transaction layer, predecessor, or state bounds differ",
    )
    require(
        next_inputs.get("transaction_index") == next_transaction
        and next_inputs.get("layer_index") == layer_index
        and next_inputs.get("transaction_position") == transaction_position
        and next_inputs.get("required_result") == descriptor.get("required_result")
        and next_inputs.get("template_input_binding_sha256")
        == descriptor.get("template_input_binding_sha256")
        == descriptor.get("input_binding_sha256"),
        "readiness transaction, layer, or input-binding bounds differ",
    )
    fixture_key = f"layer{layer_index}_fixture_manifest"
    kv_key = f"layer{layer_index}_position2_kv_parent"
    require(
        next_inputs.get(fixture_key)
        == descriptor.get("inputs", {}).get("fixture_manifest")
        and next_inputs.get(kv_key)
        == descriptor.get("inputs", {}).get("position2_kv_parent"),
        "next-layer fixture or K/V binding differs from invocation",
    )
    _, rope_rows = validate_fixture(
        next_inputs[fixture_key],
        layer_index=layer_index,
        transaction_position=transaction_position,
    )
    kv_parent = authenticate_record(next_inputs[kv_key], "next-layer K/V parent")
    require(
        Path(kv_parent["path"]).name
        == f"position{transaction_position:03d}.state",
        "next-layer K/V cache parent position differs",
    )

    state_key = f"checkpoint{parent_transaction:03d}_hidden_kv_state"
    state_record = next_inputs.get(state_key)
    require(
        isinstance(state_record, dict),
        f"{state_key} binding is missing",
    )
    parent_hidden_sha256 = validate_parent_receipt(
        parent_record,
        state_record,
        parent_transaction_index=parent_transaction,
        layer_index=layer_index,
        output_state_position=output_state_position,
    )
    token_id, feedback_sha256 = validate_selected_token_feedback(
        document,
        generation_manifest=generation_manifest,
        ledger=ledger,
        transaction_position=transaction_position,
    )
    return {
        "schema_version": 1,
        "kind": "ace3_next_position_traversal_preflight",
        "status": "PASS",
        "inputs_status": "READY_INPUTS",
        "launch_status": "NOT_READY",
        "launch_authorized": False,
        "execution_performed": False,
        "runtime_identity": runtime_identity,
        "generation": generation,
        "cursor": next_transaction,
        "parent_checkpoint_index": parent_transaction,
        "transaction_index": next_transaction,
        "layer_index": layer_index,
        "transaction_position": transaction_position,
        "output_state_position": output_state_position,
        "selected_token_id": token_id,
        "selected_token_feedback_semantic_sha256": feedback_sha256,
        "parent_hidden_semantic_sha256": parent_hidden_sha256,
        "authenticated_fixture_tensor_count": len(
            required_tensor_names(layer_index)
        ),
        "authenticated_rope_coefficient_rows": rope_rows,
        "transaction_count_bound": TRANSACTION_COUNT,
        "model_layer_count_bound": MODEL_LAYERS,
    }


def validate_preflight(path: Path) -> dict[str, Any]:
    return validate_preflight_document(load_json(path, "runtime readiness"))


def canonical_json(document: object) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate_preflight(args.readiness.resolve())
        payload = canonical_json(report)
        if args.output is not None:
            write_new(args.output.resolve(), payload)
        sys.stdout.buffer.write(payload)
    except (OSError, PreflightError) as error:
        print(f"NEXT_POSITION_PREFLIGHT_NOT_READY {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
