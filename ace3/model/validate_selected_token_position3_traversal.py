#!/usr/bin/env python3
"""Execute and validate an authenticated position-3 RTL traversal."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from safetensors import safe_open

from accept_position3_traversal_launch import (
    LaunchAcceptanceError,
    PREFLIGHT_NAME,
    TRAVERSAL_OPERATION,
    validate as validate_launch_preflight,
)
from official_model24_dialogue import _load_model
from official_model24_next_token import _layer_tensor_names, _tensor_record
from prepare_position3_continuation import (
    CHECKPOINT_SHA256,
    HIDDEN_SIZE,
    LAYER_COUNT,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    POSITION0_TOKEN_ID,
    POSITION1_TOKEN_ID,
    POSITION2_TOKEN_ID,
    POSITION3,
    PreparationError,
    canonical_json,
    file_record,
    load_json,
)
from validate_selected_token_position2_traversal import (
    ContinuationError as Position2ContinuationError,
    execute_transaction,
    hidden_payload,
    load_hidden_bits,
    materialize_transaction_vectors,
    parse_natural_terminal,
    run_logged,
    selected_primary_layer_step as _primary_layer_step,
    semantic_hidden_sha256,
)
from qwen2_rope_oracle import qwen2_coefficient


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREFLIGHT = (
    ROOT
    / "build/model24_selected_token_position3_launch_acceptance"
    / "preflight/launch_preflight.json"
)
DEFAULT_OUTPUT = ROOT / "build/model24_selected_token_position3/evidence.json"
NUMERIC_PROFILE = (
    "native asymmetric packed INT4 AWQ W4A16 G128, no qzero plus-one "
    "adjustment, FP16 activations and FP16 K/V"
)
RECHECK_CONDITION = (
    "recheck when "
    "build/model24_selected_token_position3_launch_acceptance/preflight/"
    "launch_preflight.json exists with status ACCEPTED and passes fresh "
    "validation of its exact embedding, token history, source/artifact "
    "hashes, 24-layer FP16 K/V parentage, and "
    "selected-token-position3-full-traversal operation"
)
CONSUMED_SOURCE_PATHS = {
    "position3_traversal_executor": (
        "ace3/model/validate_selected_token_position3_traversal.py"
    ),
    "focused_tests": (
        "ace3/model/tests/test_validate_selected_token_position3_traversal.py"
    ),
    "contract": "ace3/contracts/selected_token_position3_traversal.json",
    "position3_launch_acceptor": (
        "ace3/model/accept_position3_traversal_launch.py"
    ),
    "position3_package_preparer": (
        "ace3/model/prepare_position3_continuation.py"
    ),
    "position2_traversal_executor": (
        "ace3/model/validate_selected_token_position2_traversal.py"
    ),
    "kv_and_integer_oracle": "ace3/model/official_model24_dialogue.py",
    "official_model24_next_token": "ace3/model/official_model24_next_token.py",
    "official_single_decoder_layer": (
        "ace3/model/official_single_decoder_layer.py"
    ),
    "model24_execution_oracle": "ace3/model/model24_execution_oracle.py",
    "model24_oracle": "ace3/model/model24_oracle.py",
    "attention_oracle": "ace3/model/attention_oracle.py",
    "fp16_adaptation_oracle": "ace3/model/fp16_adaptation_oracle.py",
    "awq_bit_oracle": "ace3/model/awq_bit_oracle.py",
    "qwen2_rope_oracle": "ace3/model/qwen2_rope_oracle.py",
    "decoder_rtl": "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
    "fp16_silu_rtl": "ace3/rtl/ace3_fp16_silu_gate_core.sv",
    "decoder_verilator_harness": (
        "ace3/tb/ace3_decoder_layer0_token_engine_main.cpp"
    ),
    "trace_capture_policy": "ace3/tb/ace3_layer0_trace_capture_policy.h",
    "makefile": "Makefile",
}

LaunchValidator = Callable[..., dict[str, Any]]
TraversalExecutor = Callable[[Path, Mapping[str, Any]], dict[str, Any]]
class TraversalError(RuntimeError):
    """Raised when position-3 traversal inputs or evidence are invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TraversalError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def record_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": record.get("path"),
        "bytes": record.get("bytes"),
        "sha256": record.get("sha256"),
    }


def authenticate_record(
    record: Any,
    label: str,
    *,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    require(isinstance(record, dict), f"{label} record is malformed")
    require(isinstance(record.get("path"), str), f"{label} path is missing")
    path = Path(record["path"])
    require(path.is_file(), f"{label} is missing: {path}")
    actual = file_record(path)
    require(record_fields(record) == actual, f"{label} content binding mismatch")
    if expected_path is not None:
        require(
            actual["path"] == str(expected_path.resolve()),
            f"{label} substituted path binding",
        )
    return actual


def authenticate_record_tree(value: Any, label: str) -> None:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            authenticate_record(value, label)
            return
        for name, child in value.items():
            authenticate_record_tree(child, f"{label} {name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            authenticate_record_tree(child, f"{label} {index}")


def source_records(
    repository_root: Path = ROOT,
    paths: Mapping[str, str] = CONSUMED_SOURCE_PATHS,
) -> dict[str, dict[str, Any]]:
    return {
        label: file_record(repository_root / relative)
        for label, relative in paths.items()
    }


def require_source_bindings(
    stored: Any,
    repository_root: Path = ROOT,
    paths: Mapping[str, str] = CONSUMED_SOURCE_PATHS,
) -> None:
    require(isinstance(stored, dict), "consumed source closure is missing")
    actual = source_records(repository_root, paths)
    require(set(stored) == set(actual), "consumed source closure mismatch")
    for label, record in actual.items():
        require(
            stored.get(label) == record,
            f"{label} source binding mismatch",
        )


def require_ordered_layers(layers: Sequence[Mapping[str, Any]]) -> None:
    require(
        [layer.get("layer_index") for layer in layers]
        == list(range(LAYER_COUNT)),
        "layer records are not ordered 0 through 23",
    )


def authenticate_preflight(
    preflight_path: Path,
    *,
    launch_validator: LaunchValidator = validate_launch_preflight,
    launch_validation_kwargs: Mapping[str, Any] | None = None,
    expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
) -> dict[str, Any]:
    require(
        preflight_path.is_file(),
        f"position-3 launch preflight is missing: {preflight_path}",
    )
    require(
        preflight_path.name == PREFLIGHT_NAME
        and {path.name for path in preflight_path.parent.iterdir()}
        == {PREFLIGHT_NAME},
        "position-3 launch preflight artifact closure mismatch",
    )
    preflight_record = file_record(preflight_path)
    document = load_json(preflight_path)
    accepted = document.get("accepted_package")
    require(isinstance(accepted, dict), "accepted package binding is missing")
    manifest_record = authenticate_record(
        accepted.get("launch_manifest"),
        "accepted launch manifest",
    )
    manifest_path = Path(manifest_record["path"])
    require(
        manifest_path.name == "launch_manifest.json",
        "accepted launch manifest substituted path binding",
    )
    package_dir = manifest_path.parent
    kwargs = dict(launch_validation_kwargs or {})
    fresh = launch_validator(package_dir, preflight_path.parent, **kwargs)
    require(
        fresh == document,
        "stored position-3 launch preflight is stale or malformed",
    )
    require(
        file_record(preflight_path) == preflight_record,
        "position-3 launch preflight changed during validation",
    )

    manifest = load_json(manifest_path)
    model = document.get("model")
    position3_input = manifest.get("position3_input")
    validated = document.get("validated_inputs")
    authority = document.get("launch_authority")
    require(
        document.get("schema_version") == 1
        and document.get("kind")
        == "ace3_position3_traversal_inert_launch_authority_preflight"
        and document.get("status") == "ACCEPTED",
        "position-3 launch preflight identity mismatch",
    )
    require(
        model
        == {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "numeric_profile": NUMERIC_PROFILE,
        },
        "position-3 launch preflight model binding mismatch",
    )
    require(
        isinstance(position3_input, dict)
        and isinstance(validated, dict)
        and isinstance(authority, dict),
        "position-3 launch preflight bindings are incomplete",
    )
    selected_token_id = position3_input.get("selected_token_id")
    prompt_history = [POSITION0_TOKEN_ID, POSITION1_TOKEN_ID, POSITION2_TOKEN_ID]
    require(
        isinstance(selected_token_id, int)
        and 0 <= selected_token_id < 151936
        and validated.get("position") == POSITION3
        and validated.get("selected_token_id") == selected_token_id
        and validated.get("selected_logit_f16_bits")
        == position3_input.get("selected_logit_f16_bits")
        and validated.get("prompt_token_history") == prompt_history
        and validated.get("traversal_token_history")
        == [*prompt_history, selected_token_id],
        "position-3 embedding or token-history binding mismatch",
    )
    require(
        validated.get("kv_source_position") == 2
        and validated.get("kv_target_position") == POSITION3
        and validated.get("kv_parent_layers") == LAYER_COUNT,
        "position-3 K/V parentage header mismatch",
    )
    require(
        authority
        == {
            "operation": TRAVERSAL_OPERATION,
            "state": "INERT_PREFLIGHT_ONLY",
            "all_required_bindings_valid": True,
            "execution_authority": False,
            "durable_launch_authority": False,
            "execution_performed": False,
            "traversal_output_created": False,
        },
        "position-3 traversal operation or authority mismatch",
    )
    require(
        accepted.get("parents") == manifest.get("parents")
        and validated.get("package_source_bindings")
        == manifest.get("source_bindings")
        and validated.get("package_artifact_bindings")
        == manifest.get("consumed_artifacts"),
        "position-3 package closure binding mismatch",
    )
    embedding_record = authenticate_record(
        accepted.get("position3_embedding"),
        "accepted position-3 embedding",
        expected_path=package_dir / "position3_input.hex",
    )
    require(
        record_fields(position3_input.get("embedding", {})) == embedding_record,
        "position-3 embedding manifest binding mismatch",
    )
    authenticate_record_tree(
        validated["package_source_bindings"], "package sources"
    )
    authenticate_record_tree(
        validated["package_artifact_bindings"], "package artifacts"
    )
    authenticate_record_tree(
        document.get("acceptance_source_bindings"), "acceptance sources"
    )

    kv_parentage = manifest.get("layer_kv_parentage")
    layers = kv_parentage.get("layers") if isinstance(kv_parentage, dict) else None
    require(
        isinstance(layers, list) and len(layers) == LAYER_COUNT,
        "position-3 K/V parent layer closure mismatch",
    )
    require_ordered_layers(layers)
    embedding_bits = load_hidden_bits(Path(embedding_record["path"]))
    require(
        sha256_bytes(np.asarray(embedding_bits, dtype="<u2").tobytes())
        == semantic_hidden_sha256(Path(embedding_record["path"])),
        "position-3 embedding semantic binding mismatch",
    )
    checkpoint = validated["package_artifact_bindings"]["official_checkpoint"]
    require(
        checkpoint.get("sha256") == expected_checkpoint_sha256,
        "official checkpoint SHA-256 mismatch",
    )
    return {
        "preflight": document,
        "preflight_record": preflight_record,
        "manifest": manifest,
        "manifest_record": manifest_record,
        "embedding_bits": embedding_bits,
        "checkpoint": checkpoint,
        "selected_token_id": selected_token_id,
        "prompt_token_history": prompt_history,
        "traversal_token_history": [*prompt_history, selected_token_id],
        "kv_parentage": layers,
    }


def materialize_position3_vectors(
    checkpoint: Any,
    layer_index: int,
    hidden_bits: np.ndarray,
    vector_dir: Path,
) -> dict[str, Any]:
    vectors = materialize_transaction_vectors(
        checkpoint,
        layer_index,
        POSITION3,
        hidden_bits,
        vector_dir,
    )
    manifest_path = Path(vectors["manifest"]["path"])
    manifest = load_json(manifest_path)
    manifest["kind"] = "ace3_position3_live_transaction_vectors"
    manifest_path.write_bytes(canonical_json(manifest))
    return {**vectors, "manifest": file_record(manifest_path)}


def serialized_tensor_payload(value: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(value)
    unit = contiguous.dtype.itemsize
    payload = contiguous.tobytes()
    return "".join(
        f"{int.from_bytes(payload[offset:offset + unit], 'little'):0{unit * 2}x}\n"
        for offset in range(0, len(payload), unit)
    ).encode("ascii")


def position3_rope_payload() -> bytes:
    return "".join(
        f"{rope_position:04x}{pair:02x}{cosine:04x}{sine:04x}\n"
        for rope_position in range(POSITION3 + 1)
        for pair in range(32)
        for cosine, sine in (qwen2_coefficient(rope_position, pair),)
    ).encode("ascii")


def content_identity(record: Mapping[str, Any]) -> tuple[Any, Any]:
    return record.get("bytes"), record.get("sha256")


def require_same_content(
    stored: Mapping[str, Any],
    fresh: Mapping[str, Any],
    label: str,
) -> None:
    require(
        content_identity(stored) == content_identity(fresh),
        f"{label} replay mismatch",
    )


def normalized_file_sha256(path: Path, roots: Sequence[Path]) -> str:
    payload = path.read_bytes()
    resolved = sorted(
        {str(root.resolve()).encode() for root in roots},
        key=len,
        reverse=True,
    )
    for root in resolved:
        payload = payload.replace(root, b"<EXECUTION_ROOT>")
    return sha256_bytes(payload)


def build_canonical_layer(
    layer_index: int,
    build_root: Path,
) -> dict[str, dict[str, Any]]:
    compile_log = build_root / f"layer{layer_index:02d}-compile.log"
    compile_log.parent.mkdir(parents=True, exist_ok=True)
    run_logged(
        [
            "make",
            "--no-print-directory",
            "model24-rtl-layer-compile",
            f"MODEL24_RTL_LAYER_INDEX={layer_index}",
            "MODEL24_RTL_ACCURATE_SILU=1",
            f"MODEL24_RTL_CASCADE_DIR={build_root}",
        ],
        compile_log,
    )
    binary = (
        build_root
        / f"compiled/layer{layer_index}/obj_dir"
        / "Vace3_decoder_layer0_token_engine"
    )
    require(binary.is_file(), f"layer {layer_index} canonical binary is missing")
    return {
        "binary": file_record(binary),
        "compile_log": file_record(compile_log),
    }


def validate_vector_closure(
    layer_index: int,
    vectors: Mapping[str, Any],
    previous_sha256: str,
    checkpoint: Any,
    vector_dir: Path,
) -> None:
    manifest_record = authenticate_record(
        vectors.get("manifest"),
        f"layer {layer_index} vector manifest",
        expected_path=vector_dir / "manifest.json",
    )
    input_record = authenticate_record(
        vectors.get("input"),
        f"layer {layer_index} vector input",
        expected_path=vector_dir / "inputs.hex",
    )
    rope_record = authenticate_record(
        vectors.get("rope_coefficients"),
        f"layer {layer_index} vector rope_coefficients",
        expected_path=vector_dir / "rope_coefficients.hex",
    )
    tensors = vectors.get("tensors")
    expected_names = _layer_tensor_names(layer_index)
    require(
        isinstance(tensors, list) and len(tensors) == len(expected_names),
        f"layer {layer_index} tensor closure mismatch",
    )
    manifest = load_json(Path(manifest_record["path"]))
    require(
        set(manifest)
        == {
            "schema_version",
            "kind",
            "layer_index",
            "position",
            "input_activation_sha256",
            "input",
            "rope_coefficients",
            "tensors",
        }
        and manifest.get("schema_version") == 1
        and manifest.get("kind") == "ace3_position3_live_transaction_vectors"
        and manifest.get("layer_index") == layer_index
        and manifest.get("position") == POSITION3
        and manifest.get("input_activation_sha256") == previous_sha256
        and manifest.get("input") == input_record
        and manifest.get("rope_coefficients") == rope_record
        and manifest.get("tensors") == tensors,
        f"layer {layer_index} vector manifest identity mismatch",
    )
    require(
        semantic_hidden_sha256(Path(input_record["path"])) == previous_sha256,
        f"layer {layer_index} input vector semantic mismatch",
    )
    rope_payload = position3_rope_payload()
    require(
        content_identity(rope_record)
        == (len(rope_payload), sha256_bytes(rope_payload)),
        f"layer {layer_index} RoPE vector binding mismatch",
    )
    actual_names = []
    for tensor_index, (tensor, expected_name) in enumerate(
        zip(tensors, expected_names, strict=True)
    ):
        require(
            isinstance(tensor, dict)
            and set(tensor) == {"checkpoint_tensor", "serialized"}
            and isinstance(tensor.get("checkpoint_tensor"), dict),
            f"layer {layer_index} tensor {tensor_index} malformed",
        )
        checkpoint_tensor = tensor["checkpoint_tensor"]
        actual_names.append(checkpoint_tensor.get("name"))
        require(
            checkpoint_tensor.get("name") == expected_name,
            f"layer {layer_index} tensor inventory mismatch",
        )
        value = np.asarray(checkpoint.get_tensor(expected_name))
        require(
            checkpoint_tensor == _tensor_record(expected_name, value),
            f"layer {layer_index} tensor {expected_name} checkpoint binding mismatch",
        )
        serialized = authenticate_record(
            tensor.get("serialized"),
            f"layer {layer_index} tensor {expected_name}",
            expected_path=(
                vector_dir
                / "tensors"
                / (
                    f"layer{layer_index}_"
                    f"{expected_name.removeprefix(f'model.layers.{layer_index}.').replace('.', '_')}."
                    f"{'fp16le.bin' if value.dtype.itemsize == 2 else 'i32le.bin'}.hex"
                )
            ),
        )
        expected_payload = serialized_tensor_payload(value)
        require(
            content_identity(serialized)
            == (len(expected_payload), sha256_bytes(expected_payload)),
            f"layer {layer_index} tensor {expected_name} serialization mismatch",
        )
    require(
        actual_names == list(expected_names)
        and len(set(actual_names)) == len(expected_names),
        f"layer {layer_index} tensor inventory mismatch",
    )


def seed_integer_oracle(
    authenticated: Mapping[str, Any],
) -> list[Any]:
    checkpoint_path = Path(authenticated["checkpoint"]["path"])
    embeddings, _, _, states = _load_model(checkpoint_path)
    parent_layers = authenticated["kv_parentage"]
    for position, token_id in enumerate(authenticated["prompt_token_history"]):
        hidden = np.asarray(embeddings[token_id], dtype="<f2").view("<u2")
        for layer_index, state in enumerate(states):
            hidden = _primary_layer_step(
                state,
                hidden.reshape(1, HIDDEN_SIZE),
                position,
            )[0]
            if position == 2:
                parent_hidden = Path(
                    parent_layers[layer_index]["position2_output_hidden"]["path"]
                )
                require(
                    sha256_bytes(np.asarray(hidden, dtype="<u2").tobytes())
                    == semantic_hidden_sha256(parent_hidden),
                    f"layer {layer_index} position-2 parent differs from "
                    "independent integer oracle",
                )
    return states


def comparison_record(
    actual_bits: np.ndarray,
    expected_bits: np.ndarray,
) -> dict[str, Any]:
    require(
        np.array_equal(actual_bits, expected_bits),
        "RTL output differs from independent exact integer oracle",
    )
    return {
        "implementation": "independent ACE-3 integer W4A16 oracle",
        "inter_layer_boundary": "binary16 round after every RTL layer",
        "exact_integer_oracle_output_sha256": sha256_bytes(
            np.asarray(expected_bits, dtype="<u2").tobytes()
        ),
        "rtl_matches_exact_integer_oracle": True,
        "integer_mismatches": 0,
    }


def execute_live_traversal(
    live_root: Path,
    authenticated: Mapping[str, Any],
) -> dict[str, Any]:
    require(not live_root.exists(), "position-3 traversal output already exists")
    live_root.mkdir(parents=True)
    states = seed_integer_oracle(authenticated)
    hidden_bits = np.asarray(authenticated["embedding_bits"], dtype="<u2")
    layers = []
    checkpoint_path = Path(authenticated["checkpoint"]["path"])
    with safe_open(checkpoint_path, framework="np") as checkpoint:
        for layer_index, (state, parentage) in enumerate(
            zip(states, authenticated["kv_parentage"], strict=True)
        ):
            layer_dir = live_root / f"layer{layer_index:02d}"
            vector_dir = layer_dir / "vectors"
            vectors = materialize_position3_vectors(
                checkpoint,
                layer_index,
                hidden_bits,
                vector_dir,
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
            binary = (
                live_root
                / f"compiled/layer{layer_index}/obj_dir"
                / "Vace3_decoder_layer0_token_engine"
            )
            require(binary.is_file(), f"layer {layer_index} RTL binary is missing")
            expected_bits = _primary_layer_step(
                state,
                hidden_bits.reshape(1, HIDDEN_SIZE),
                POSITION3,
            )[0]
            transaction, actual_bits = execute_transaction(
                binary,
                layer_index,
                POSITION3,
                hidden_bits,
                vectors,
                vector_dir,
                layer_dir / "position003",
                layer_dir / "position004.state",
                Path(parentage["position2_output_state"]["path"]),
            )
            comparison = comparison_record(actual_bits, expected_bits)
            report_path = layer_dir / "position003/comparison.json"
            report_path.write_bytes(canonical_json(comparison))
            layers.append(
                {
                    **transaction,
                    "authenticated_kv_parentage": parentage,
                    "predecessor_state": parentage["position2_output_state"],
                    "live_binary": file_record(binary),
                    "compile_log": file_record(compile_log),
                    "independent_oracle_comparison": comparison,
                    "comparison_report": file_record(report_path),
                }
            )
            hidden_bits = actual_bits
    return {
        "status": "COMPLETE",
        "execution": "current-worktree compiled Verilator RTL",
        "operation": TRAVERSAL_OPERATION,
        "selected_token_id": authenticated["selected_token_id"],
        "position": POSITION3,
        "prompt_token_history": authenticated["prompt_token_history"],
        "traversal_token_history": authenticated["traversal_token_history"],
        "layer_order": list(range(LAYER_COUNT)),
        "natural_terminal_layers": LAYER_COUNT,
        "parent_preflight": authenticated["preflight_record"],
        "parent_launch_manifest": authenticated["manifest_record"],
        "layers": layers,
        "post_layer23": {
            "hidden_sha256": layers[-1]["output"]["semantic_sha256"],
            "natural_terminal": True,
            "independent_integer_oracle_match": True,
        },
    }


def replay_stored_layers(
    layers: Sequence[Mapping[str, Any]],
    authenticated: Mapping[str, Any],
    replay_root: Path,
) -> list[dict[str, Any]]:
    states = seed_integer_oracle(authenticated)
    hidden_bits = np.asarray(authenticated["embedding_bits"], dtype="<u2")
    compared = []
    stored_root = Path(layers[0]["live_binary"]["path"]).parents[3]
    build_root = replay_root / "build"
    for layer_index, (layer, state) in enumerate(zip(layers, states, strict=True)):
        canonical = build_canonical_layer(layer_index, build_root)
        require_same_content(
            layer["live_binary"],
            canonical["binary"],
            f"layer {layer_index} canonical binary",
        )
        require(
            normalized_file_sha256(
                Path(layer["compile_log"]["path"]),
                [stored_root],
            )
            == normalized_file_sha256(
                Path(canonical["compile_log"]["path"]),
                [build_root],
            ),
            f"layer {layer_index} compile log replay mismatch",
        )
        transaction, actual_bits = execute_transaction(
            Path(canonical["binary"]["path"]),
            layer_index,
            POSITION3,
            hidden_bits,
            layer["vectors"],
            Path(layer["vectors"]["manifest"]["path"]).parent,
            replay_root / f"layer{layer_index:02d}/position003",
            replay_root / f"layer{layer_index:02d}/position004.state",
            Path(
                authenticated["kv_parentage"][layer_index][
                    "position2_output_state"
                ]["path"]
            ),
        )
        require(
            transaction["input"] == layer["input"],
            f"layer {layer_index} replay input mismatch",
        )
        for label in ("output", "output_state"):
            require_same_content(
                layer[label],
                transaction[label],
                f"layer {layer_index} {label}",
            )
        require(
            normalized_file_sha256(
                Path(layer["simulation_log"]["path"]),
                [stored_root],
            )
            == normalized_file_sha256(
                Path(transaction["simulation_log"]["path"]),
                [replay_root],
            ),
            f"layer {layer_index} simulation log replay mismatch",
        )
        for label in ("terminal", "trace"):
            require_same_content(
                layer["raw"][label],
                transaction["raw"][label],
                f"layer {layer_index} {label}",
            )
        require(
            all(
                layer["raw"].get(name) == transaction["raw"].get(name)
                for name in ("trace_count", "final_count", "done_count")
            ),
            f"layer {layer_index} terminal count replay mismatch",
        )
        stored_bits = load_hidden_bits(Path(layer["output"]["path"]))
        require(
            np.array_equal(actual_bits, stored_bits),
            f"layer {layer_index} replayed output mismatch",
        )
        expected_bits = _primary_layer_step(
            state,
            hidden_bits.reshape(1, HIDDEN_SIZE),
            POSITION3,
        )[0]
        comparison = comparison_record(actual_bits, expected_bits)
        require(
            layer.get("independent_oracle_comparison") == comparison,
            f"layer {layer_index} independent oracle record mismatch",
        )
        require(
            load_json(Path(layer["comparison_report"]["path"])) == comparison,
            f"layer {layer_index} independent oracle report mismatch",
        )
        compared.append(dict(layer))
        hidden_bits = actual_bits
    return compared


def validate_traversal(
    traversal: Mapping[str, Any],
    authenticated: Mapping[str, Any],
    traversal_root: Path,
) -> dict[str, Any]:
    require(
        traversal.get("status") == "COMPLETE"
        and traversal.get("execution")
        == "current-worktree compiled Verilator RTL"
        and traversal.get("operation") == TRAVERSAL_OPERATION
        and traversal.get("selected_token_id")
        == authenticated["selected_token_id"]
        and traversal.get("position") == POSITION3
        and traversal.get("prompt_token_history")
        == authenticated["prompt_token_history"]
        and traversal.get("traversal_token_history")
        == authenticated["traversal_token_history"]
        and traversal.get("layer_order") == list(range(LAYER_COUNT))
        and traversal.get("natural_terminal_layers") == LAYER_COUNT
        and traversal.get("parent_preflight")
        == authenticated["preflight_record"]
        and traversal.get("parent_launch_manifest")
        == authenticated["manifest_record"],
        "position-3 traversal identity mismatch",
    )
    layers = traversal.get("layers")
    require(isinstance(layers, list), "position-3 traversal layers are missing")
    require_ordered_layers(layers)
    previous_sha256 = sha256_bytes(
        np.asarray(authenticated["embedding_bits"], dtype="<u2").tobytes()
    )
    checkpoint_path = Path(authenticated["checkpoint"]["path"])
    with safe_open(checkpoint_path, framework="np") as checkpoint:
        for layer_index, (layer, parentage) in enumerate(
            zip(layers, authenticated["kv_parentage"], strict=True)
        ):
            require(
                layer.get("position") == POSITION3
                and layer.get("input", {}).get("sha256") == previous_sha256,
                f"layer {layer_index} activation lineage mismatch",
            )
            require(
                layer.get("authenticated_kv_parentage") == parentage
                and layer.get("predecessor_state")
                == parentage["position2_output_state"],
                f"layer {layer_index} K/V parentage mismatch",
            )
            authenticate_record_tree(parentage, f"layer {layer_index} K/V parent")
            for label in (
                "output",
                "output_state",
                "live_binary",
                "compile_log",
                "simulation_log",
                "comparison_report",
            ):
                authenticate_record(
                    layer.get(label), f"layer {layer_index} {label}"
                )
            vectors = layer.get("vectors")
            require(
                isinstance(vectors, dict),
                f"layer {layer_index} vectors missing",
            )
            validate_vector_closure(
                layer_index,
                vectors,
                previous_sha256,
                checkpoint,
                traversal_root / f"layer{layer_index:02d}/vectors",
            )
            raw = layer.get("raw")
            require(
                isinstance(raw, dict),
                f"layer {layer_index} raw evidence missing",
            )
            authenticate_record(
                raw.get("terminal"), f"layer {layer_index} terminal"
            )
            authenticate_record(raw.get("trace"), f"layer {layer_index} trace")
            counts = parse_natural_terminal(
                Path(raw["terminal"]["path"]), layer_index, POSITION3
            )
            require(
                all(raw.get(name) == value for name, value in counts.items()),
                f"layer {layer_index} natural terminal count mismatch",
            )
            output_sha256 = semantic_hidden_sha256(Path(layer["output"]["path"]))
            require(
                layer["output"].get("semantic_sha256") == output_sha256,
                f"layer {layer_index} output semantic binding mismatch",
            )
            comparison = layer.get("independent_oracle_comparison")
            require(
                isinstance(comparison, dict)
                and comparison.get("rtl_matches_exact_integer_oracle") is True
                and comparison.get("integer_mismatches") == 0
                and comparison.get("exact_integer_oracle_output_sha256")
                == output_sha256,
                f"layer {layer_index} independent oracle binding mismatch",
            )
            require(
                load_json(Path(layer["comparison_report"]["path"])) == comparison,
                f"layer {layer_index} comparison report mismatch",
            )
            previous_sha256 = output_sha256
    with TemporaryDirectory(prefix="ace3-position3-replay-") as directory:
        compared = replay_stored_layers(
            layers,
            authenticated,
            Path(directory),
        )
    require(compared == layers, "stored position-3 comparisons are stale")
    require(
        traversal.get("post_layer23")
        == {
            "hidden_sha256": previous_sha256,
            "natural_terminal": True,
            "independent_integer_oracle_match": True,
        },
        "position-3 post-layer-23 binding mismatch",
    )
    return dict(traversal)


def assemble_evidence(
    authenticated: Mapping[str, Any],
    traversal: Mapping[str, Any],
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ace3_selected_token_position3_continuation_evidence",
        "status": "COMPLETE",
        "model": authenticated["preflight"]["model"],
        "authenticated_launch_preflight": authenticated["preflight_record"],
        "position3_input": {
            "position": POSITION3,
            "selected_token_id": authenticated["selected_token_id"],
            "prompt_token_history": authenticated["prompt_token_history"],
            "traversal_token_history": authenticated["traversal_token_history"],
            "embedding": authenticated["preflight"]["accepted_package"][
                "position3_embedding"
            ],
        },
        "current_continuation_attempt": dict(traversal),
        "consumed_sources": dict(sources),
        "claim_boundary": {
            "demonstrated": (
                "authenticated position-3 selected-token embedding through "
                "24 current-worktree Verilator decoder-layer transactions "
                "with ordered FP16 K/V parentage and exact integer-oracle "
                "agreement"
            ),
            "dialogue": "not claimed",
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
            "latency": "not measured",
            "throughput": "not measured",
        },
    }


def generate(
    preflight_path: Path,
    output: Path,
    *,
    launch_validator: LaunchValidator = validate_launch_preflight,
    launch_validation_kwargs: Mapping[str, Any] | None = None,
    expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
    traversal_executor: TraversalExecutor = execute_live_traversal,
    repository_root: Path = ROOT,
    consumed_source_paths: Mapping[str, str] = CONSUMED_SOURCE_PATHS,
) -> dict[str, Any]:
    require(not output.exists(), f"position-3 evidence already exists: {output}")
    authenticated = authenticate_preflight(
        preflight_path,
        launch_validator=launch_validator,
        launch_validation_kwargs=launch_validation_kwargs,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    traversal = traversal_executor(output.parent / "traversal", authenticated)
    validated = validate_traversal(
        traversal,
        authenticated,
        output.parent / "traversal",
    )
    document = assemble_evidence(
        authenticated,
        validated,
        source_records(repository_root, consumed_source_paths),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(document))
    return document


def validate(
    preflight_path: Path,
    output: Path,
    *,
    launch_validator: LaunchValidator = validate_launch_preflight,
    launch_validation_kwargs: Mapping[str, Any] | None = None,
    expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
    repository_root: Path = ROOT,
    consumed_source_paths: Mapping[str, str] = CONSUMED_SOURCE_PATHS,
) -> dict[str, Any]:
    authenticated = authenticate_preflight(
        preflight_path,
        launch_validator=launch_validator,
        launch_validation_kwargs=launch_validation_kwargs,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    stored = load_json(output)
    require_source_bindings(
        stored.get("consumed_sources"),
        repository_root,
        consumed_source_paths,
    )
    traversal = stored.get("current_continuation_attempt")
    require(
        isinstance(traversal, dict),
        "stored position-3 continuation attempt is missing",
    )
    validated = validate_traversal(
        traversal,
        authenticated,
        output.parent / "traversal",
    )
    fresh = assemble_evidence(
        authenticated,
        validated,
        source_records(repository_root, consumed_source_paths),
    )
    require(stored == fresh, "stored position-3 traversal evidence is stale")
    return stored


def print_summary(document: Mapping[str, Any], output: Path) -> None:
    traversal = document["current_continuation_attempt"]
    print(
        "MODEL24_SELECTED_TOKEN_POSITION3_TRAVERSAL_PASS "
        f"status={document['status']} "
        f"selected_token={document['position3_input']['selected_token_id']} "
        f"operation={traversal['operation']} "
        f"layers={len(traversal['layers'])} layer_order=0..23 "
        "fp16_kv_parent_layers=24 natural_terminals=24 "
        "integer_mismatches=0 current_traversal=current_worktree_verilator "
        "synthesis=not_run ppa=not_measured fpga=not_run "
        "latency=not_measured throughput=not_measured "
        f"evidence={output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("generate", "validate"))
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    preflight = args.preflight.resolve()
    output = args.output.resolve()
    try:
        document = (
            generate(preflight, output)
            if args.operation == "generate"
            else validate(preflight, output)
        )
    except (
        TraversalError,
        LaunchAcceptanceError,
        PreparationError,
        Position2ContinuationError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        raise SystemExit(
            f"POSITION3_TRAVERSAL_NOT_READY {error}; {RECHECK_CONDITION}"
        ) from error
    print_summary(document, output)


if __name__ == "__main__":
    main()
