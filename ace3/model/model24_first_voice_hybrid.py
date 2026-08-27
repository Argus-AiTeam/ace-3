#!/usr/bin/env python3
"""Bounded transactional Hybrid RTL dialogue driver for the fixed Model24."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors import safe_open

from fp16_adaptation_oracle import rmsnorm
from model24_execution_oracle import (
    EOS_TOKEN_ID,
    FIXED_CHAT_MESSAGES,
    FIXED_CHAT_SERIALIZATION,
    FIXED_CHAT_TOKEN_IDS,
    Model24TokenDecisionHost,
    TENSOR_MAP_SHA256,
    authenticate_tokenizer,
    exact_tied_lm_head_logits,
    indexed_layer_tensor_records,
    serialize_chat_prompt,
)
from model24_oracle import (
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    authenticate_checkpoint,
)
from official_model24_dialogue import (
    LOGITS_ABSOLUTE_TOLERANCE,
    _load_model,
    _reference_layer_step,
    generation_stop_reason,
)
from official_model24_next_token import (
    HIDDEN_SIZE,
    LAYER_COUNT,
    TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
)
from official_single_decoder_layer import (
    _bits_to_f16,
    _canonical_bytes,
    _f16_to_bits,
    _torch_rmsnorm,
)
from qwen2_rope_oracle import qwen2_coefficient

KIND = "ace3_model24_first_voice_hybrid_execution"
CONTRACT_RELATIVE_PATH = "ace3/contracts/model24_first_voice_hybrid.json"
BUILD_MANIFEST_KIND = "ace3_model24_first_voice_verilator_build"
COMPACT_LAYER_MANIFEST_KIND = "ace3_model24_first_voice_compact_layer"
STATE_KIND = "ace3_model24_first_voice_layer_state"
STATE_SCHEMA_VERSION = 2
STATE_ENVELOPE_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "model_binding",
        "build_manifest_sha256",
        "binary_sha256",
        "layer_index",
        "cache_slot",
        "next_position",
        "parent_state_sha256",
        "parent_envelope_sha256",
        "input_activation_sha256",
        "output_hidden_sha256",
        "state",
    }
)
STATE_HASH_RECORD_KEYS = frozenset({"bytes", "sha256"})
TRUSTED_TIP_KIND = "ace3_model24_first_voice_trusted_tip"
TRUSTED_TIP_SCHEMA_VERSION = 1
TRUSTED_TIP_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "model_binding",
        "build_manifest_sha256",
        "binary_sha256",
        "layer_index",
        "cache_slot",
        "next_position",
        "envelope",
        "state",
        "input_activation_sha256",
        "output_hidden_sha256",
    }
)
TRUSTED_TIP_CHECKPOINT_KIND = "ace3_model24_first_voice_trusted_tips_checkpoint"
TRUSTED_TIP_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "model_binding",
        "build_manifest_sha256",
        "tips",
    }
)
MODEL_BINDING_KEYS = frozenset({"repository", "revision", "checkpoint_sha256"})
MAX_POSITIONS = 128
DEFAULT_MAX_NEW_TOKENS = 4
MINIMUM_MAX_NEW_TOKENS = 2
RTL_BINARY_NAME = "Vace3_decoder_layer0_token_engine"
RTL_TOP_MODULE = "ace3_decoder_layer0_token_engine"
COMPACT_SELF_TEST_MARKER = "DECODER_LAYER_TOKEN_ENGINE_COMPACT_BUILD_PASS"

RTL_BUILD_SOURCES = (
    "ace3/rtl/ace3_fp16_fixed.sv",
    "ace3/rtl/ace3_q47_48_to_f16_rne.sv",
    "ace3/rtl/ace3_awq_w4a16_g128_dot_lane.sv",
    "ace3/rtl/ace3_awq_w4a16_projection_engine.sv",
    "ace3/rtl/ace3_fp16_rmsnorm_core.sv",
    "ace3/rtl/ace3_fp16_residual_add_core.sv",
    "ace3/rtl/ace3_fp16_silu_gate_core.sv",
    "ace3/rtl/ace3_qwen2_rope_pair.sv",
    "ace3/rtl/ace3_fp16_kv_cache.sv",
    "ace3/rtl/ace3_attention_score_core.sv",
    "ace3/rtl/ace3_attention_softmax_core.sv",
    "ace3/rtl/ace3_attention_value_core.sv",
    "ace3/rtl/ace3_decoder_qzeros_address.sv",
    "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
    "ace3/tb/ace3_layer0_trace_capture_policy.h",
    "ace3/tb/ace3_decoder_layer0_token_engine_main.cpp",
)


class HybridRtlError(RuntimeError):
    """Fail-closed execution error with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = {} if details is None else dict(details)


def require(
    condition: bool,
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    if not condition:
        raise HybridRtlError(code, message, details)


def canonical_json(document: Any) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while payload := stream.read(1024 * 1024):
            digest.update(payload)
            size += len(payload)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_hash_record(record: Any, label: str) -> dict[str, Any]:
    require(
        type(record) is dict
        and set(record) == STATE_HASH_RECORD_KEYS
        and type(record.get("bytes")) is int
        and record["bytes"] >= 0
        and _is_sha256(record.get("sha256")),
        "stale_rtl_state",
        f"{label} hash record schema mismatch",
    )
    return record


def _validate_model_binding(binding: Any, label: str) -> dict[str, Any]:
    require(
        type(binding) is dict
        and set(binding) == MODEL_BINDING_KEYS
        and type(binding.get("repository")) is str
        and type(binding.get("revision")) is str
        and _is_sha256(binding.get("checkpoint_sha256"))
        and binding
        == {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "stale_rtl_state",
        f"{label} model binding schema mismatch",
    )
    return binding


def _validate_state_envelope_schema(
    envelope: Any,
    label: str,
) -> dict[str, Any]:
    require(
        type(envelope) is dict
        and set(envelope) == STATE_ENVELOPE_KEYS
        and type(envelope.get("schema_version")) is int
        and envelope["schema_version"] == STATE_SCHEMA_VERSION
        and type(envelope.get("kind")) is str
        and envelope["kind"] == STATE_KIND
        and type(envelope.get("layer_index")) is int
        and 0 <= envelope["layer_index"] < LAYER_COUNT
        and type(envelope.get("cache_slot")) is int
        and envelope["cache_slot"] == 0
        and type(envelope.get("next_position")) is int
        and envelope["next_position"] > 0
        and _is_sha256(envelope.get("build_manifest_sha256"))
        and _is_sha256(envelope.get("binary_sha256"))
        and (
            envelope.get("parent_state_sha256") is None
            or _is_sha256(envelope.get("parent_state_sha256"))
        )
        and (
            envelope.get("parent_envelope_sha256") is None
            or _is_sha256(envelope.get("parent_envelope_sha256"))
        )
        and _is_sha256(envelope.get("input_activation_sha256"))
        and _is_sha256(envelope.get("output_hidden_sha256")),
        "stale_rtl_state",
        f"{label} state envelope schema mismatch",
    )
    _validate_model_binding(envelope.get("model_binding"), label)
    _validate_hash_record(envelope.get("state"), f"{label} state")
    return envelope


def _validate_trusted_tip(tip: Any, label: str) -> dict[str, Any]:
    require(
        type(tip) is dict
        and set(tip) == TRUSTED_TIP_KEYS
        and type(tip.get("schema_version")) is int
        and tip["schema_version"] == TRUSTED_TIP_SCHEMA_VERSION
        and type(tip.get("kind")) is str
        and tip["kind"] == TRUSTED_TIP_KIND
        and type(tip.get("layer_index")) is int
        and 0 <= tip["layer_index"] < LAYER_COUNT
        and type(tip.get("cache_slot")) is int
        and tip["cache_slot"] == 0
        and type(tip.get("next_position")) is int
        and tip["next_position"] > 0
        and _is_sha256(tip.get("build_manifest_sha256"))
        and _is_sha256(tip.get("binary_sha256"))
        and _is_sha256(tip.get("input_activation_sha256"))
        and _is_sha256(tip.get("output_hidden_sha256")),
        "stale_rtl_state",
        f"{label} trusted tip schema mismatch",
    )
    _validate_model_binding(tip.get("model_binding"), label)
    _validate_hash_record(tip.get("envelope"), f"{label} envelope")
    _validate_hash_record(tip.get("state"), f"{label} state")
    return tip


def state_tip_commitment(
    envelope: Mapping[str, Any],
    envelope_payload: bytes,
) -> dict[str, Any]:
    validated = _validate_state_envelope_schema(envelope, "new")
    require(
        envelope_payload == canonical_json(validated),
        "stale_rtl_state",
        "new state envelope payload is not canonical",
    )
    return {
        "schema_version": TRUSTED_TIP_SCHEMA_VERSION,
        "kind": TRUSTED_TIP_KIND,
        "model_binding": dict(validated["model_binding"]),
        "build_manifest_sha256": validated["build_manifest_sha256"],
        "binary_sha256": validated["binary_sha256"],
        "layer_index": validated["layer_index"],
        "cache_slot": validated["cache_slot"],
        "next_position": validated["next_position"],
        "envelope": {
            "bytes": len(envelope_payload),
            "sha256": sha256_bytes(envelope_payload),
        },
        "state": dict(validated["state"]),
        "input_activation_sha256": validated["input_activation_sha256"],
        "output_hidden_sha256": validated["output_hidden_sha256"],
    }


def write_trusted_tips_checkpoint(
    path: Path,
    tips: Mapping[int, Mapping[str, Any]],
    *,
    build_manifest_sha256: str,
) -> dict[str, Any]:
    require(
        _is_sha256(build_manifest_sha256),
        "stale_rtl_state",
        "trusted tips checkpoint build binding schema mismatch",
    )
    ordered = []
    for layer_index in sorted(tips):
        require(
            type(layer_index) is int,
            "stale_rtl_state",
            "trusted tips checkpoint layer key schema mismatch",
        )
        tip = _validate_trusted_tip(
            tips[layer_index], f"layer {layer_index} checkpoint"
        )
        require(
            tip["layer_index"] == layer_index
            and tip["build_manifest_sha256"] == build_manifest_sha256,
            "stale_rtl_state",
            f"layer {layer_index} trusted tip checkpoint binding mismatch",
        )
        ordered.append(tip)
    document = {
        "schema_version": 1,
        "kind": TRUSTED_TIP_CHECKPOINT_KIND,
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "build_manifest_sha256": build_manifest_sha256,
        "tips": ordered,
    }
    write_json(path, document)
    return hash_file(path)


def load_trusted_tips_checkpoint(
    path: Path,
    expected_checkpoint: Mapping[str, Any] | None,
    *,
    build_manifest_sha256: str,
) -> dict[int, dict[str, Any]]:
    require(
        expected_checkpoint is not None,
        "stale_rtl_state",
        "trusted tips checkpoint requires an independently authenticated digest",
    )
    expected = _validate_hash_record(
        expected_checkpoint, "expected trusted tips checkpoint"
    )
    require(
        path.is_file() and hash_file(path) == expected,
        "stale_rtl_state",
        "trusted tips checkpoint digest mismatch",
    )
    payload = path.read_bytes()
    document = load_json(path, "trusted tips checkpoint")
    require(
        payload == canonical_json(document)
        and type(document) is dict
        and set(document) == TRUSTED_TIP_CHECKPOINT_KEYS
        and type(document.get("schema_version")) is int
        and document["schema_version"] == 1
        and type(document.get("kind")) is str
        and document["kind"] == TRUSTED_TIP_CHECKPOINT_KIND
        and _is_sha256(document.get("build_manifest_sha256"))
        and document["build_manifest_sha256"] == build_manifest_sha256
        and type(document.get("tips")) is list,
        "stale_rtl_state",
        "trusted tips checkpoint schema or build binding mismatch",
    )
    _validate_model_binding(document.get("model_binding"), "checkpoint")
    tips: dict[int, dict[str, Any]] = {}
    for raw_tip in document["tips"]:
        tip = _validate_trusted_tip(raw_tip, "checkpoint")
        layer_index = tip["layer_index"]
        require(
            layer_index not in tips
            and tip["build_manifest_sha256"] == build_manifest_sha256,
            "stale_rtl_state",
            f"layer {layer_index} trusted tip checkpoint is duplicate or stale",
        )
        tips[layer_index] = tip
    return tips


def artifact_path(path: Path, repository_root: Path) -> str:
    try:
        return str(path.relative_to(repository_root))
    except ValueError:
        return str(path)


def load_json(path: Path, label: str) -> dict[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(
                key not in result,
                "ambiguous_json",
                f"{label} has duplicate key {key}",
            )
            result[key] = value
        return result

    try:
        document = json.loads(path.read_bytes(), object_pairs_hook=object_hook)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HybridRtlError(
            "invalid_json", f"{label} is not valid JSON: {error}"
        ) from error
    require(
        isinstance(document, dict),
        "invalid_json",
        f"{label} root must be an object",
    )
    return document


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(canonical_json(document))
    os.replace(partial, path)


def plan_execution(
    prompt_ids: Sequence[int],
    max_new_tokens: int,
) -> dict[str, int]:
    require(
        bool(prompt_ids)
        and all(type(token_id) is int and token_id >= 0 for token_id in prompt_ids),
        "invalid_prompt",
        "prompt token IDs must be a non-empty integer sequence",
    )
    require(
        type(max_new_tokens) is int
        and max_new_tokens >= MINIMUM_MAX_NEW_TOKENS,
        "invalid_generation_bound",
        f"max_new_tokens must be at least {MINIMUM_MAX_NEW_TOKENS}",
    )
    represented = len(prompt_ids) + max_new_tokens - 1
    require(
        represented <= MAX_POSITIONS,
        "rtl_context_capacity_exceeded",
        (
            f"fixed RTL cache can represent {MAX_POSITIONS} positions, but the "
            f"{len(prompt_ids)}-token prompt plus worst-case generated-token "
            f"feedback requires {represented}"
        ),
        {
            "cache_positions": MAX_POSITIONS,
            "prompt_positions": len(prompt_ids),
            "requested_new_tokens": max_new_tokens,
            "required_represented_positions": represented,
        },
    )
    return {
        "prompt_positions": len(prompt_ids),
        "maximum_generated_feedback_positions": max_new_tokens - 1,
        "maximum_represented_positions": represented,
        "cache_positions": MAX_POSITIONS,
    }


def contract_binding(repository_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = repository_root / CONTRACT_RELATIVE_PATH
    payload = path.read_bytes()
    contract = load_json(path, "First Voice contract")
    require(
        contract.get("kind") == "ace3_model24_first_voice_hybrid_contract"
        and type(contract.get("schema_version")) is int
        and contract.get("schema_version") == 2,
        "contract_mismatch",
        "First Voice contract schema mismatch",
    )
    model = contract.get("model_binding", {})
    require(
        model
        == {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_bytes": CHECKPOINT_SIZE,
        },
        "contract_mismatch",
        "First Voice fixed model binding mismatch",
    )
    numeric = contract.get("numeric_profile", {})
    require(
        numeric.get("weights") == "native asymmetric packed INT4 AWQ"
        and numeric.get("group_size") == 128
        and numeric.get("qzero_adjustment") == "none"
        and numeric.get("activations") == "FP16"
        and numeric.get("kv") == "FP16",
        "contract_mismatch",
        "First Voice native AWQ numeric profile mismatch",
    )
    rtl = contract.get("rtl", {})
    require(
        rtl.get("indexed_layers") == LAYER_COUNT
        and rtl.get("cache_positions") == MAX_POSITIONS
        and rtl.get("required_verilator_option") == "--savable",
        "contract_mismatch",
        "First Voice RTL capability binding mismatch",
    )
    return contract, {
        "path": CONTRACT_RELATIVE_PATH,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def compact_layer_dir(compiled_dir: Path, layer_index: int) -> Path:
    return compiled_dir / f"layer{layer_index}"


def binary_path(compiled_dir: Path, layer_index: int) -> Path:
    return compact_layer_dir(compiled_dir, layer_index) / "bin" / RTL_BINARY_NAME


def compact_layer_manifest_path(compiled_dir: Path, layer_index: int) -> Path:
    return compact_layer_dir(compiled_dir, layer_index) / "layer_manifest.json"


def source_hashes(repository_root: Path) -> dict[str, str]:
    return {
        relative: sha256_bytes((repository_root / relative).read_bytes())
        for relative in RTL_BUILD_SOURCES
    }


def compact_build_configuration(
    layer_index: int,
    verilator_version: str,
) -> dict[str, Any]:
    require(
        type(layer_index) is int and 0 <= layer_index < LAYER_COUNT,
        "invalid_layer_index",
        f"compact RTL layer index must be in [0,{LAYER_COUNT - 1}]",
    )
    require(
        isinstance(verilator_version, str) and bool(verilator_version.strip()),
        "invalid_build_configuration",
        "Verilator version binding must be non-empty",
    )
    source_arguments = [
        relative
        for relative in RTL_BUILD_SOURCES
        if relative.endswith((".sv", ".cpp"))
    ]
    return {
        "verilator_version": verilator_version.strip(),
        "top_module": RTL_TOP_MODULE,
        "options": ["--cc", "--exe", "--build", "--savable", "--Wall", "-Wno-fatal"],
        "parameters": {
            "LAYER_INDEX": layer_index,
            "ACCURATE_SILU": 1,
        },
        "source_arguments": source_arguments,
        "strip_arguments": ["--strip-all"],
    }


def _compact_layer_document(
    repository_root: Path,
    compiled_dir: Path,
    layer_index: int,
    verilator_version: str,
    binary: Path,
) -> dict[str, Any]:
    _, contract_record = contract_binding(repository_root)
    configuration = compact_build_configuration(layer_index, verilator_version)
    return {
        "schema_version": 1,
        "kind": COMPACT_LAYER_MANIFEST_KIND,
        "layer_index": layer_index,
        "contract": contract_record,
        "sources": source_hashes(repository_root),
        "configuration": configuration,
        "configuration_sha256": sha256_bytes(canonical_json(configuration)),
        "binary": {
            "path": str(binary_path(compiled_dir, layer_index).relative_to(compiled_dir)),
            **hash_file(binary),
        },
    }


def write_compact_layer_manifest(
    repository_root: Path,
    compiled_dir: Path,
    layer_index: int,
    verilator_version: str,
) -> dict[str, Any]:
    binary = binary_path(compiled_dir, layer_index)
    require(
        binary.is_file() and os.access(binary, os.X_OK),
        "compiled_layer_missing",
        f"compact indexed RTL layer {layer_index} binary is missing",
    )
    document = _compact_layer_document(
        repository_root,
        compiled_dir,
        layer_index,
        verilator_version,
        binary,
    )
    write_json(compact_layer_manifest_path(compiled_dir, layer_index), document)
    return document


def authenticate_compact_layer(
    repository_root: Path,
    compiled_dir: Path,
    layer_index: int,
) -> tuple[dict[str, Any], str]:
    manifest_path = compact_layer_manifest_path(compiled_dir, layer_index)
    require(
        manifest_path.is_file(),
        "compiled_layer_missing",
        f"compact indexed RTL layer {layer_index} manifest is missing",
        {"path": str(manifest_path)},
    )
    payload = manifest_path.read_bytes()
    manifest = load_json(manifest_path, "First Voice compact layer manifest")
    require(
        payload == canonical_json(manifest),
        "stale_compiled_rtl",
        f"compact RTL layer {layer_index} manifest is not canonical",
    )
    require(
        manifest.get("schema_version") == 1
        and manifest.get("kind") == COMPACT_LAYER_MANIFEST_KIND
        and manifest.get("layer_index") == layer_index,
        "stale_compiled_rtl",
        f"compact RTL layer {layer_index} manifest identity mismatch",
    )
    _, contract_record = contract_binding(repository_root)
    require(
        manifest.get("contract") == contract_record
        and manifest.get("sources") == source_hashes(repository_root),
        "stale_compiled_rtl",
        f"compact RTL layer {layer_index} source or contract binding is stale",
    )
    configuration = manifest.get("configuration")
    require(
        isinstance(configuration, dict)
        and isinstance(configuration.get("verilator_version"), str),
        "stale_compiled_rtl",
        f"compact RTL layer {layer_index} configuration is missing",
    )
    require(
        configuration
        == compact_build_configuration(
            layer_index,
            configuration["verilator_version"],
        )
        and manifest.get("configuration_sha256")
        == sha256_bytes(canonical_json(configuration)),
        "stale_compiled_rtl",
        f"compact RTL layer {layer_index} configuration hash mismatch",
    )
    binary = binary_path(compiled_dir, layer_index)
    binary_record = manifest.get("binary")
    require(
        isinstance(binary_record, dict)
        and binary_record.get("path") == str(binary.relative_to(compiled_dir))
        and binary.is_file()
        and os.access(binary, os.X_OK)
        and hash_file(binary)
        == {key: binary_record.get(key) for key in ("bytes", "sha256")},
        "stale_compiled_rtl",
        f"compact RTL layer {layer_index} binary hash mismatch",
    )
    return manifest, sha256_bytes(payload)


def build_compact_layer(
    repository_root: Path,
    compiled_dir: Path,
    layer_index: int,
    temporary_mdir: Path,
    *,
    verilator: str,
    strip: str,
) -> dict[str, Any]:
    require(
        type(layer_index) is int and 0 <= layer_index < LAYER_COUNT,
        "invalid_layer_index",
        f"compact RTL layer index must be in [0,{LAYER_COUNT - 1}]",
    )
    repository_root = repository_root.resolve(strict=True)
    compiled_dir = compiled_dir.resolve()
    temporary_mdir = temporary_mdir.resolve()
    require(
        temporary_mdir.name == "compact_mdir"
        and temporary_mdir not in (repository_root, compiled_dir)
        and temporary_mdir not in repository_root.parents
        and temporary_mdir not in compiled_dir.parents,
        "invalid_temporary_mdir",
        "temporary Mdir must be a dedicated compact_mdir path",
    )
    final_layer_dir = compact_layer_dir(compiled_dir, layer_index)
    staging_layer_dir = compiled_dir / f".layer{layer_index}.partial"
    staging_binary = staging_layer_dir / "bin" / RTL_BINARY_NAME
    shutil.rmtree(staging_layer_dir, ignore_errors=True)
    shutil.rmtree(temporary_mdir, ignore_errors=True)
    compiled_dir.mkdir(parents=True, exist_ok=True)
    temporary_mdir.mkdir(parents=True)
    try:
        try:
            version_result = subprocess.run(
                [verilator, "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            verilator_version = (
                version_result.stdout or version_result.stderr
            ).strip()
            configuration = compact_build_configuration(
                layer_index,
                verilator_version,
            )
            command = [verilator, *configuration["options"]]
            command.extend(
                [
                    f"-GLAYER_INDEX={layer_index}",
                    f"-GACCURATE_SILU={configuration['parameters']['ACCURATE_SILU']}",
                    "--top-module",
                    RTL_TOP_MODULE,
                    "--Mdir",
                    str(temporary_mdir),
                ]
            )
            command.extend(
                str(repository_root / relative)
                for relative in configuration["source_arguments"]
            )
            subprocess.run(command, cwd=repository_root, check=True)
            built_binary = temporary_mdir / RTL_BINARY_NAME
            require(
                built_binary.is_file() and os.access(built_binary, os.X_OK),
                "compact_build_failed",
                f"Verilator did not produce compact RTL layer {layer_index} executable",
            )
            staging_binary.parent.mkdir(parents=True)
            shutil.copyfile(built_binary, staging_binary)
            staging_binary.chmod(0o755)
            subprocess.run(
                [strip, *configuration["strip_arguments"], str(staging_binary)],
                check=True,
            )
            document = _compact_layer_document(
                repository_root,
                compiled_dir,
                layer_index,
                verilator_version,
                staging_binary,
            )
            write_json(staging_layer_dir / "layer_manifest.json", document)
        finally:
            shutil.rmtree(temporary_mdir, ignore_errors=True)

        self_test = subprocess.run(
            [
                str(staging_binary),
                "--compact-build-self-test",
                "--layer-index",
                str(layer_index),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        require(
            f"{COMPACT_SELF_TEST_MARKER} layer_index={layer_index}"
            in self_test.stdout,
            "compact_build_failed",
            f"compact RTL layer {layer_index} executable self-test marker is missing",
        )
        if final_layer_dir.exists():
            shutil.rmtree(final_layer_dir)
        os.replace(staging_layer_dir, final_layer_dir)
        accepted, _ = authenticate_compact_layer(
            repository_root,
            compiled_dir,
            layer_index,
        )
        return accepted
    except HybridRtlError:
        shutil.rmtree(staging_layer_dir, ignore_errors=True)
        raise
    except (OSError, subprocess.SubprocessError) as error:
        shutil.rmtree(staging_layer_dir, ignore_errors=True)
        raise HybridRtlError(
            "compact_build_failed",
            f"compact RTL layer {layer_index} build failed: {error}",
        ) from error


def bind_compiled(repository_root: Path, compiled_dir: Path) -> dict[str, Any]:
    _, contract_record = contract_binding(repository_root)
    layers = []
    for layer_index in range(LAYER_COUNT):
        layer_manifest, layer_manifest_sha256 = authenticate_compact_layer(
            repository_root,
            compiled_dir,
            layer_index,
        )
        manifest_path = compact_layer_manifest_path(compiled_dir, layer_index)
        layers.append(
            {
                "layer_index": layer_index,
                "manifest": {
                    "path": str(manifest_path.relative_to(compiled_dir)),
                    **hash_file(manifest_path),
                },
                "manifest_sha256": layer_manifest_sha256,
                "configuration_sha256": layer_manifest["configuration_sha256"],
                "binary": layer_manifest["binary"],
            }
        )
    document = {
        "schema_version": 2,
        "kind": BUILD_MANIFEST_KIND,
        "verilator_savable": True,
        "compact_layout": True,
        "contract": contract_record,
        "sources": source_hashes(repository_root),
        "layers": layers,
    }
    write_json(compiled_dir / "build_manifest.json", document)
    return document


def authenticate_build(
    repository_root: Path,
    compiled_dir: Path,
    contract_record: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    manifest_path = compiled_dir / "build_manifest.json"
    require(
        manifest_path.is_file(),
        "compiled_layer_missing",
        "compiled RTL build manifest is missing",
        {"path": str(manifest_path)},
    )
    manifest_payload = manifest_path.read_bytes()
    manifest = load_json(manifest_path, "First Voice build manifest")
    require(
        manifest_payload == canonical_json(manifest)
        and manifest.get("schema_version") == 2
        and manifest.get("kind") == BUILD_MANIFEST_KIND
        and manifest.get("verilator_savable") is True
        and manifest.get("compact_layout") is True,
        "stale_compiled_rtl",
        "compiled RTL build manifest schema, layout, or --savable binding mismatch",
    )
    require(
        manifest.get("contract") == dict(contract_record)
        and manifest.get("sources") == source_hashes(repository_root),
        "stale_compiled_rtl",
        "compiled RTL source or contract binding is stale",
    )
    layers = manifest.get("layers")
    require(
        isinstance(layers, list)
        and [record.get("layer_index") for record in layers]
        == list(range(LAYER_COUNT)),
        "stale_compiled_rtl",
        "compiled RTL layer inventory mismatch",
    )
    for record in layers:
        layer_index = record["layer_index"]
        layer_manifest_path = compact_layer_manifest_path(
            compiled_dir,
            layer_index,
        )
        require(
            layer_manifest_path.is_file()
            and record.get("manifest")
            == {
                "path": str(layer_manifest_path.relative_to(compiled_dir)),
                **hash_file(layer_manifest_path),
            },
            "stale_compiled_rtl",
            f"compiled RTL layer {layer_index} manifest hash mismatch",
        )
        layer_manifest, layer_manifest_sha256 = authenticate_compact_layer(
            repository_root,
            compiled_dir,
            layer_index,
        )
        require(
            record.get("manifest_sha256") == layer_manifest_sha256
            and record.get("configuration_sha256")
            == layer_manifest["configuration_sha256"]
            and record.get("binary") == layer_manifest["binary"],
            "stale_compiled_rtl",
            f"compiled RTL layer {layer_index} binding mismatch",
        )
    return manifest, sha256_bytes(manifest_payload)


def compiled_binary_hashes(
    build_manifest: Mapping[str, Any],
) -> dict[int, str]:
    return {
        record["layer_index"]: record["binary"]["sha256"]
        for record in build_manifest["layers"]
    }


def state_record_paths(
    states_dir: Path,
    layer_index: int,
    next_position: int,
) -> tuple[Path, Path]:
    record_dir = (
        states_dir
        / f"layer{layer_index:02d}"
        / f"position{next_position:03d}"
    )
    return record_dir / "state", record_dir / "envelope.json"


def _retained_hidden_sha256(path: Path, label: str) -> str:
    require(
        path.is_file(),
        "stale_rtl_state",
        f"retained {label} is missing",
        {"path": str(path)},
    )
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise HybridRtlError(
            "stale_rtl_state",
            f"retained {label} is unreadable: {error}",
        ) from error
    require(
        len(lines) == HIDDEN_SIZE,
        "stale_rtl_state",
        f"retained {label} record count mismatch",
    )
    values = np.empty(HIDDEN_SIZE, dtype="<u2")
    for expected_index, line in enumerate(lines):
        try:
            valid = (
                len(line) == 10
                and line[:2] == "00"
                and int(line[2:6], 16) == expected_index
            )
            value = int(line[6:10], 16) if valid else 0
        except ValueError:
            valid = False
            value = 0
        require(
            valid,
            "stale_rtl_state",
            f"retained {label} ordering mismatch",
        )
        values[expected_index] = value
    return sha256_bytes(_canonical_bytes(values))


def _validate_retained_state_record(
    *,
    states_dir: Path,
    runtime_dir: Path,
    layer_index: int,
    next_position: int,
    build_manifest_sha256: str,
    binary_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    state_path, envelope_path = state_record_paths(
        states_dir,
        layer_index,
        next_position,
    )
    require(
        state_path.is_file() and envelope_path.is_file(),
        "stale_rtl_state",
        f"layer {layer_index} retained state position {next_position} is missing",
    )
    envelope_payload = envelope_path.read_bytes()
    envelope = load_json(envelope_path, f"layer {layer_index} state envelope")
    require(
        envelope_payload == canonical_json(envelope),
        "stale_rtl_state",
        f"layer {layer_index} state envelope is not canonical",
    )
    _validate_state_envelope_schema(envelope, f"layer {layer_index}")
    require(
        envelope.get("model_binding")
        == {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "stale_rtl_state",
        f"layer {layer_index} state checkpoint binding mismatch",
    )
    require(
        envelope.get("layer_index") == layer_index
        and envelope.get("cache_slot") == 0
        and envelope.get("next_position") == next_position
        and envelope.get("build_manifest_sha256") == build_manifest_sha256
        and envelope.get("binary_sha256") == binary_sha256,
        "stale_rtl_state",
        f"layer {layer_index} state owner or causal position mismatch",
    )
    state_record = envelope.get("state")
    require(
        isinstance(state_record, dict)
        and set(state_record) == STATE_HASH_RECORD_KEYS
        and state_record == hash_file(state_path),
        "stale_rtl_state",
        f"layer {layer_index} opaque Verilator state hash mismatch",
    )

    transition_position = next_position - 1
    require(
        transition_position >= 0,
        "stale_rtl_state",
        f"layer {layer_index} state has no producing transition",
    )
    transaction_dir = (
        runtime_dir
        / f"position{transition_position:03d}"
        / f"layer{layer_index:02d}"
    )
    require(
        envelope.get("input_activation_sha256")
        == _retained_hidden_sha256(
            transaction_dir / "inputs.hex",
            f"layer {layer_index} position {transition_position} activation",
        )
        and envelope.get("output_hidden_sha256")
        == _retained_hidden_sha256(
            transaction_dir / "raw" / "final.hex",
            f"layer {layer_index} position {transition_position} output",
        ),
        "stale_rtl_state",
        f"layer {layer_index} activation or output lineage mismatch",
    )

    if next_position == 1:
        require(
            envelope.get("parent_state_sha256") is None
            and envelope.get("parent_envelope_sha256") is None,
            "stale_rtl_state",
            f"layer {layer_index} genesis state has a predecessor",
        )
    else:
        parent, parent_payload = _validate_retained_state_record(
            states_dir=states_dir,
            runtime_dir=runtime_dir,
            layer_index=layer_index,
            next_position=next_position - 1,
            build_manifest_sha256=build_manifest_sha256,
            binary_sha256=binary_sha256,
        )
        require(
            envelope.get("parent_state_sha256") == parent["state"]["sha256"]
            and envelope.get("parent_envelope_sha256")
            == sha256_bytes(parent_payload),
            "stale_rtl_state",
            f"layer {layer_index} predecessor lineage mismatch",
        )
    return envelope, envelope_payload


def validate_state_envelope(
    envelope_path: Path,
    state_path: Path,
    *,
    states_dir: Path,
    runtime_dir: Path,
    layer_index: int,
    next_position: int,
    build_manifest_sha256: str,
    binary_sha256: str,
    expected_tip: Mapping[str, Any] | None,
) -> dict[str, Any]:
    require(
        expected_tip is not None,
        "stale_rtl_state",
        f"layer {layer_index} restore has no trusted chain-tip commitment",
    )
    tip = _validate_trusted_tip(expected_tip, f"layer {layer_index}")
    require(
        tip["layer_index"] == layer_index
        and tip["cache_slot"] == 0
        and tip["next_position"] == next_position
        and tip["build_manifest_sha256"] == build_manifest_sha256
        and tip["binary_sha256"] == binary_sha256,
        "stale_rtl_state",
        f"layer {layer_index} trusted chain-tip owner or position mismatch",
    )
    expected_state_path, expected_envelope_path = state_record_paths(
        states_dir,
        layer_index,
        next_position,
    )
    require(
        state_path == expected_state_path
        and envelope_path == expected_envelope_path,
        "stale_rtl_state",
        f"layer {layer_index} state record path mismatch",
    )
    envelope, envelope_payload = _validate_retained_state_record(
        states_dir=states_dir,
        runtime_dir=runtime_dir,
        layer_index=layer_index,
        next_position=next_position,
        build_manifest_sha256=build_manifest_sha256,
        binary_sha256=binary_sha256,
    )
    require(
        tip["envelope"]
        == {
            "bytes": len(envelope_payload),
            "sha256": sha256_bytes(envelope_payload),
        }
        and tip["state"] == hash_file(state_path)
        and tip["input_activation_sha256"]
        == envelope["input_activation_sha256"]
        and tip["output_hidden_sha256"] == envelope["output_hidden_sha256"],
        "stale_rtl_state",
        f"layer {layer_index} trusted chain-tip digest mismatch",
    )
    return envelope


def authenticate_fixed_inputs(checkpoint_path: Path, tokenizer_dir: Path) -> Any:
    try:
        authenticate_checkpoint(checkpoint_path)
    except Exception as error:
        raise HybridRtlError(
            "checkpoint_authentication_failed",
            f"official checkpoint authentication failed: {error}",
        ) from error
    try:
        return authenticate_tokenizer(tokenizer_dir)
    except Exception as error:
        raise HybridRtlError(
            "tokenizer_authentication_failed",
            f"official tokenizer authentication failed: {error}",
        ) from error


def _authenticate_tensor_map(tensor_map_path: Path) -> dict[str, Any]:
    tensor_map_payload = tensor_map_path.read_bytes()
    require(
        sha256_bytes(tensor_map_payload) == TENSOR_MAP_SHA256,
        "tensor_map_mismatch",
        "reviewed Model24 tensor map SHA256 mismatch",
    )
    return load_json(tensor_map_path, "Model24 tensor map")


def _serialize_layer_tensors(
    checkpoint_path: Path,
    tensor_map: Mapping[str, Any],
    layer_index: int,
    vector_dir: Path,
    manifest_dir: Path,
) -> dict[str, Any]:
    if vector_dir.exists():
        shutil.rmtree(vector_dir)
    tensor_dir = vector_dir / "tensors"
    tensor_dir.mkdir(parents=True)
    records = []
    prefix = f"model.layers.{layer_index}."
    with safe_open(checkpoint_path, framework="np") as checkpoint:
        for record in indexed_layer_tensor_records(tensor_map, layer_index):
            name = record["name"]
            value = np.asarray(checkpoint.get_tensor(name))
            dtype = {"F16": "<f2", "I32": "<i4"}.get(record["dtype"])
            require(
                dtype is not None and list(value.shape) == record["shape"],
                "checkpoint_tensor_mismatch",
                f"{name} dtype or shape mismatch",
            )
            raw = np.ascontiguousarray(value, dtype=dtype).tobytes()
            require(
                len(raw) == record["byte_length"],
                "checkpoint_tensor_mismatch",
                f"{name} byte length mismatch",
            )
            unit = 2 if record["dtype"] == "F16" else 4
            suffix = name.removeprefix(prefix)
            serialized = (
                f"layer{layer_index}_{suffix.replace('.', '_')}."
                f"{'fp16le.bin' if unit == 2 else 'i32le.bin'}.hex"
            )
            path = tensor_dir / serialized
            path.write_text(
                "".join(
                    f"{int.from_bytes(raw[index:index + unit], 'little'):0{unit * 2}x}\n"
                    for index in range(0, len(raw), unit)
                ),
                encoding="ascii",
            )
            records.append(
                {
                    "name": name,
                    "serialized": f"tensors/{serialized}",
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                }
            )
    manifest = {
        "schema_version": 1,
        "layer_index": layer_index,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "tensors": records,
    }
    write_json(vector_dir / "manifest.json", manifest)
    persistent_manifest = manifest_dir / f"layer{layer_index:02d}.json"
    if persistent_manifest.exists():
        require(
            load_json(persistent_manifest, f"layer {layer_index} tensor manifest")
            == manifest,
            "tensor_manifest_mismatch",
            f"layer {layer_index} tensor manifest changed during traversal",
        )
    else:
        write_json(persistent_manifest, manifest)
    return {
        "layer_index": layer_index,
        "manifest": hash_file(persistent_manifest),
        "tensor_count": len(records),
        "serialized_workspace_bytes": sum(path.stat().st_size for path in tensor_dir.iterdir()),
    }


def _compact_trace(path: Path, repository_root: Path) -> dict[str, Any]:
    raw_record = hash_file(path)
    archive = path.with_name(path.name + ".gz")
    partial = archive.with_name(archive.name + ".partial")
    with path.open("rb") as source, partial.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
            shutil.copyfileobj(source, compressed)
    os.replace(partial, archive)
    digest = hashlib.sha256()
    restored_bytes = 0
    with gzip.open(archive, "rb") as restored:
        while payload := restored.read(1024 * 1024):
            digest.update(payload)
            restored_bytes += len(payload)
    require(
        restored_bytes == raw_record["bytes"]
        and digest.hexdigest() == raw_record["sha256"],
        "trace_compaction_failed",
        "compacted RTL trace does not restore to its authenticated source",
    )
    path.unlink()
    return {
        **raw_record,
        "storage": {
            "path": artifact_path(archive, repository_root),
            "compression": "gzip",
            "mtime": 0,
            **hash_file(archive),
        },
    }


def _write_transaction_inputs(
    transaction_dir: Path,
    hidden_bits: np.ndarray,
    position: int,
) -> tuple[Path, Path]:
    require(
        hidden_bits.shape == (HIDDEN_SIZE,),
        "activation_geometry_mismatch",
        "RTL transaction hidden width mismatch",
    )
    activation_path = transaction_dir / "inputs.hex"
    activation_path.write_text(
        "".join(
            f"00{index:04x}{int(value):04x}\n"
            for index, value in enumerate(hidden_bits)
        ),
        encoding="ascii",
    )
    rope_path = transaction_dir / "rope_coefficients.hex"
    rope_path.write_text(
        "".join(
            f"{position:04x}{pair:02x}{cosine:04x}{sine:04x}\n"
            for pair in range(32)
            for cosine, sine in (qwen2_coefficient(position, pair),)
        ),
        encoding="ascii",
    )
    return activation_path, rope_path


def _parse_final(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="ascii").splitlines()
    require(
        len(lines) == HIDDEN_SIZE,
        "rtl_transaction_failed",
        "RTL transaction final record count mismatch",
    )
    values = np.empty(HIDDEN_SIZE, dtype="<u2")
    for expected_index, line in enumerate(lines):
        require(
            len(line) == 10
            and line[:2] == "00"
            and int(line[2:6], 16) == expected_index,
            "rtl_transaction_failed",
            "RTL transaction final stream ordering mismatch",
        )
        values[expected_index] = int(line[6:10], 16)
    return values


def _run_transaction(
    *,
    repository_root: Path,
    compiled_dir: Path,
    vector_dir: Path,
    runtime_dir: Path,
    states_dir: Path,
    layer_index: int,
    position: int,
    hidden_bits: np.ndarray,
    build_manifest_sha256: str,
    binary_sha256: str,
    trusted_tips: dict[int, dict[str, Any]],
    restore_build_manifest_sha256: str | None = None,
    restore_binary_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    transaction_dir = runtime_dir / f"position{position:03d}" / f"layer{layer_index:02d}"
    raw_dir = transaction_dir / "raw"
    raw_dir.mkdir(parents=True)
    activation_path, rope_path = _write_transaction_inputs(
        transaction_dir, hidden_bits, position
    )
    state_path, envelope_path = state_record_paths(
        states_dir,
        layer_index,
        position,
    )
    previous = None
    if position:
        previous = validate_state_envelope(
            envelope_path,
            state_path,
            states_dir=states_dir,
            runtime_dir=runtime_dir,
            layer_index=layer_index,
            next_position=position,
            build_manifest_sha256=(
                restore_build_manifest_sha256 or build_manifest_sha256
            ),
            binary_sha256=restore_binary_sha256 or binary_sha256,
            expected_tip=trusted_tips.get(layer_index),
        )
    else:
        require(
            not (states_dir / f"layer{layer_index:02d}").exists(),
            "stale_rtl_state",
            f"unexpected initial state for layer {layer_index}",
        )

    candidate = transaction_dir / "state.out"
    metadata_path = transaction_dir / "transaction.json"
    binary = binary_path(compiled_dir, layer_index)
    command = [
        str(binary),
        "--layer-index",
        str(layer_index),
        "--vector-dir",
        str(vector_dir),
        "--raw-dir",
        str(raw_dir),
        "--transaction-position",
        str(position),
        "--transaction-input",
        str(activation_path),
        "--transaction-rope",
        str(rope_path),
        "--state-out",
        str(candidate),
        "--transaction-metadata",
        str(metadata_path),
        "--progress-interval",
        "1000000",
    ]
    if position:
        command.extend(("--state-in", str(state_path)))
    log_path = transaction_dir / "run.log"
    with log_path.open("wb") as log:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    require(
        completed.returncode == 0,
        "rtl_transaction_failed",
        (
            f"indexed RTL layer {layer_index} position {position} exited "
            f"{completed.returncode}"
        ),
        {"log": artifact_path(log_path, repository_root)},
    )
    metadata = load_json(metadata_path, "RTL transaction metadata")
    require(
        metadata.get("schema_version") == 1
        and metadata.get("kind") == "ace3_decoder_verilator_transaction"
        and metadata.get("layer_index") == layer_index
        and metadata.get("position") == position
        and metadata.get("next_position") == position + 1
        and metadata.get("cache_slot") == 0
        and metadata.get("natural_terminal") is True
        and metadata.get("trace_records", 0) > 0
        and metadata.get("final_records") == HIDDEN_SIZE
        and metadata.get("done_records") == 1,
        "rtl_transaction_failed",
        "RTL transaction metadata mismatch",
    )
    terminal = (raw_dir / "terminal.txt").read_text(encoding="ascii")
    require(
        "schema=ace3_decoder_transaction_raw_v1" in terminal
        and "natural_terminal=1" in terminal
        and f"layer_index={layer_index}" in terminal
        and f"position={position}" in terminal,
        "rtl_transaction_failed",
        "RTL transaction did not produce its natural terminal",
    )
    output_bits = _parse_final(raw_dir / "final.hex")
    state_record = hash_file(candidate)
    next_state_path, next_envelope_path = state_record_paths(
        states_dir,
        layer_index,
        position + 1,
    )
    record_dir = next_state_path.parent
    staging_dir = record_dir.with_name(f".{record_dir.name}.partial")
    require(
        not record_dir.exists(),
        "stale_rtl_state",
        f"layer {layer_index} state position {position + 1} already exists",
    )
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_state_path = staging_dir / next_state_path.name
    staging_envelope_path = staging_dir / next_envelope_path.name
    staging_dir.mkdir(parents=True)
    os.replace(candidate, staging_state_path)
    envelope = {
        "schema_version": STATE_SCHEMA_VERSION,
        "kind": STATE_KIND,
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "build_manifest_sha256": build_manifest_sha256,
        "binary_sha256": binary_sha256,
        "layer_index": layer_index,
        "cache_slot": 0,
        "next_position": position + 1,
        "parent_state_sha256": (
            None if previous is None else previous["state"]["sha256"]
        ),
        "parent_envelope_sha256": (
            None if previous is None else sha256_bytes(canonical_json(previous))
        ),
        "input_activation_sha256": sha256_bytes(_canonical_bytes(hidden_bits)),
        "output_hidden_sha256": sha256_bytes(_canonical_bytes(output_bits)),
        "state": state_record,
    }
    envelope_payload = canonical_json(envelope)
    try:
        write_json(staging_envelope_path, envelope)
        os.replace(staging_dir, record_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    trusted_tips[layer_index] = state_tip_commitment(
        envelope,
        envelope_payload,
    )
    record = {
        "layer_index": layer_index,
        "position": position,
        "binary_sha256": binary_sha256,
        "input_activation_sha256": envelope["input_activation_sha256"],
        "output_hidden_sha256": envelope["output_hidden_sha256"],
        "parent_state_sha256": envelope["parent_state_sha256"],
        "state_sha256": state_record["sha256"],
        "state_bytes": state_record["bytes"],
        "trace": _compact_trace(raw_dir / "trace.hex", repository_root),
        "final": hash_file(raw_dir / "final.hex"),
        "metadata": hash_file(metadata_path),
        "natural_terminal": True,
    }
    return output_bits, record


def _comparison(bits: np.ndarray, reference: torch.Tensor) -> dict[str, Any]:
    difference = np.abs(
        _bits_to_f16(bits).astype(np.float64)
        - reference.detach().cpu().numpy()
    )
    return {
        "implementation": "PyTorch CPU float64 dequantized-AWQ Qwen2",
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
    }


def _cache_lineage(
    transactions: Sequence[Mapping[str, Any]],
    position: int,
    parent_sha256: str | None,
) -> dict[str, Any]:
    layers = [
        {
            "layer_index": record["layer_index"],
            "parent_state_sha256": record["parent_state_sha256"],
            "state_sha256": record["state_sha256"],
        }
        for record in transactions
    ]
    payload = {
        "position_count": position + 1,
        "parent_cache_sha256": parent_sha256,
        "layers": layers,
    }
    return {**payload, "cache_sha256": sha256_bytes(canonical_json(payload))}


def _decision(
    *,
    checkpoint_path: Path,
    tokenizer: Any,
    final_norm: np.ndarray,
    reference_lm_head: torch.Tensor,
    hidden_bits: np.ndarray,
    reference_hidden: torch.Tensor,
    ordinal: int,
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    norm_outputs, _, _ = rmsnorm(
        hidden_bits.tolist(),
        _f16_to_bits(final_norm).tolist(),
    )
    require(
        all(not invalid and not saturated for _, invalid, saturated in norm_outputs),
        "host_finalization_failed",
        "host final RMSNorm produced invalid or saturated output",
    )
    normalized_bits = np.asarray(
        [bits for bits, _, _ in norm_outputs],
        dtype="<u2",
    )
    logits_bits = exact_tied_lm_head_logits(
        checkpoint_path,
        normalized_bits.tolist(),
    )
    logits_values = _bits_to_f16(logits_bits).astype(np.float64)
    reference_normalized = _torch_rmsnorm(
        reference_hidden.unsqueeze(0),
        final_norm,
    )[0]
    reference_logits = (
        reference_lm_head @ reference_normalized
    ).detach().cpu().numpy()
    hidden_comparison = _comparison(hidden_bits, reference_hidden)
    logits_difference = np.abs(logits_values - reference_logits)
    logits_max = float(logits_difference.max())
    primary_argmax = int(np.argmax(logits_values))
    reference_argmax = int(np.argmax(reference_logits))
    require(
        hidden_comparison["max_abs_error"]
        <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
        "pytorch_comparison_failed",
        (
            f"decision {ordinal} terminal hidden error "
            f"{hidden_comparison['max_abs_error']} exceeds "
            f"{TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE}"
        ),
    )
    require(
        logits_max <= LOGITS_ABSOLUTE_TOLERANCE,
        "pytorch_comparison_failed",
        (
            f"decision {ordinal} logits error {logits_max} exceeds "
            f"{LOGITS_ABSOLUTE_TOLERANCE}"
        ),
    )
    require(
        primary_argmax == reference_argmax,
        "pytorch_comparison_failed",
        (
            f"decision {ordinal} argmax mismatch: RTL Hybrid "
            f"{primary_argmax}, PyTorch {reference_argmax}"
        ),
    )
    host = Model24TokenDecisionHost(tokenizer).decide(logits_bits)
    require(
        host["argmax_token_id"] == primary_argmax,
        "host_finalization_failed",
        "authenticated token host argmax mismatch",
    )
    return {
        "ordinal": ordinal,
        "decision_position": lineage["position_count"] - 1,
        "cache_lineage": dict(lineage),
        "terminal_hidden": {
            "sha256": sha256_bytes(_canonical_bytes(hidden_bits)),
            "independent_reference": {
                **hidden_comparison,
                "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
                "within_tolerance": True,
            },
        },
        "logits": {
            "dtype": "FP16",
            "vocab_size": len(logits_bits),
            "sha256": sha256_bytes(_canonical_bytes(np.asarray(logits_bits, dtype="<u2"))),
            "independent_reference": {
                "implementation": "PyTorch CPU float64 dequantized-AWQ tied head",
                "max_abs_error": logits_max,
                "mean_abs_error": float(logits_difference.mean()),
                "absolute_tolerance": LOGITS_ABSOLUTE_TOLERANCE,
                "within_tolerance": True,
            },
        },
        "token": {
            **host,
            "independent_reference_argmax_token_id": reference_argmax,
            "argmax_matches_independent_reference": True,
        },
    }


def execute(
    *,
    repository_root: Path,
    checkpoint_path: Path,
    tokenizer_dir: Path,
    tensor_map_path: Path,
    compiled_dir: Path,
    output_dir: Path,
    max_new_tokens: int,
) -> dict[str, Any]:
    require(
        not output_dir.exists(),
        "output_not_clean",
        "First Voice output directory already exists",
    )
    _, contract_record = contract_binding(repository_root)
    tokenizer = authenticate_fixed_inputs(checkpoint_path, tokenizer_dir)
    messages = [
        {"role": role, "content": content}
        for role, content in FIXED_CHAT_MESSAGES
    ]
    prompt = serialize_chat_prompt(messages)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    require(
        prompt == FIXED_CHAT_SERIALIZATION
        and prompt_ids == list(FIXED_CHAT_TOKEN_IDS),
        "tokenizer_authentication_failed",
        "fixed official chat-template tokenization mismatch",
    )
    plan = plan_execution(prompt_ids, max_new_tokens)
    build_manifest, build_manifest_sha256 = authenticate_build(
        repository_root,
        compiled_dir,
        contract_record,
    )
    binary_hashes = compiled_binary_hashes(build_manifest)

    output_dir.mkdir(parents=True)
    vector_workspace = output_dir / "vector_workspace"
    tensor_manifest_dir = output_dir / "tensor_manifests"
    runtime_dir = output_dir / "transactions"
    states_dir = output_dir / "states"
    tensor_map = _authenticate_tensor_map(tensor_map_path)
    tensor_manifests: dict[int, dict[str, Any]] = {}
    trusted_tips: dict[int, dict[str, Any]] = {}

    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    embeddings, final_norm, reference_lm_head, reference_states = _load_model(
        checkpoint_path
    )
    positions = []
    previous_cache_sha256 = None

    def represent(token_id: int, position: int, origin: Mapping[str, Any]) -> tuple[
        np.ndarray, torch.Tensor, dict[str, Any]
    ]:
        nonlocal previous_cache_sha256
        hidden_bits = _f16_to_bits(embeddings[token_id])
        reference_hidden = reference_lm_head[token_id].clone()
        transactions = []
        layer_comparisons = []
        for layer_index, reference_state in enumerate(reference_states):
            input_bits = hidden_bits
            tensor_manifest = _serialize_layer_tensors(
                checkpoint_path,
                tensor_map,
                layer_index,
                vector_workspace,
                tensor_manifest_dir,
            )
            previous_manifest = tensor_manifests.setdefault(
                layer_index, tensor_manifest
            )
            require(
                previous_manifest == tensor_manifest,
                "tensor_manifest_mismatch",
                f"layer {layer_index} tensor workspace changed during traversal",
            )
            try:
                hidden_bits, transaction = _run_transaction(
                    repository_root=repository_root,
                    compiled_dir=compiled_dir,
                    vector_dir=vector_workspace,
                    runtime_dir=runtime_dir,
                    states_dir=states_dir,
                    layer_index=layer_index,
                    position=position,
                    hidden_bits=input_bits,
                    build_manifest_sha256=build_manifest_sha256,
                    binary_sha256=binary_hashes[layer_index],
                    trusted_tips=trusted_tips,
                )
            finally:
                if vector_workspace.exists():
                    shutil.rmtree(vector_workspace)
            reference_hidden = _reference_layer_step(
                reference_state,
                reference_hidden.unsqueeze(0),
                position,
            )[0]
            transactions.append(transaction)
            layer_comparisons.append(
                {
                    "layer_index": layer_index,
                    **_comparison(hidden_bits, reference_hidden),
                }
            )
        lineage = _cache_lineage(
            transactions,
            position,
            previous_cache_sha256,
        )
        previous_cache_sha256 = lineage["cache_sha256"]
        positions.append(
            {
                "position": position,
                "token_id": token_id,
                "origin": dict(origin),
                "layers": transactions,
                "independent_layer_comparisons": layer_comparisons,
                "cache_lineage": lineage,
            }
        )
        return hidden_bits, reference_hidden, lineage

    hidden_bits: np.ndarray | None = None
    reference_hidden: torch.Tensor | None = None
    lineage: dict[str, Any] | None = None
    for position, token_id in enumerate(prompt_ids):
        hidden_bits, reference_hidden, lineage = represent(
            token_id,
            position,
            {"kind": "prompt", "prompt_ordinal": position},
        )

    require(
        hidden_bits is not None
        and reference_hidden is not None
        and lineage is not None,
        "invalid_prompt",
        "prompt execution produced no terminal state",
    )
    generated = []
    decisions = []
    stop_reason = None
    while stop_reason is None:
        ordinal = len(generated)
        record = _decision(
            checkpoint_path=checkpoint_path,
            tokenizer=tokenizer,
            final_norm=final_norm,
            reference_lm_head=reference_lm_head,
            hidden_bits=hidden_bits,
            reference_hidden=reference_hidden,
            ordinal=ordinal,
            lineage=lineage,
        )
        token_id = record["token"]["argmax_token_id"]
        generated.append(
            {
                "ordinal": ordinal,
                "token_id": token_id,
                "selected_after_position": record["decision_position"],
                "fed_back_to_rtl": False,
                "rtl_input_position": None,
            }
        )
        decisions.append(record)
        stop_reason = generation_stop_reason(
            token_id,
            len(generated),
            max_new_tokens,
        )
        if stop_reason is not None:
            break
        position = len(prompt_ids) + ordinal
        generated[-1]["fed_back_to_rtl"] = True
        generated[-1]["rtl_input_position"] = position
        hidden_bits, reference_hidden, lineage = represent(
            token_id,
            position,
            {"kind": "generated_feedback", "generated_ordinal": ordinal},
        )

    generated_ids = [record["token_id"] for record in generated]
    decoded_ids = [
        token_id for token_id in generated_ids if token_id != EOS_TOKEN_ID
    ]
    document = {
        "schema_version": 1,
        "kind": KIND,
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint": {
                "filename": checkpoint_path.name,
                "bytes": CHECKPOINT_SIZE,
                "sha256": CHECKPOINT_SHA256,
            },
        },
        "contract": contract_record,
        "compiled_rtl": {
            "build_manifest_sha256": build_manifest_sha256,
            "verilator_savable": True,
            "indexed_layers": LAYER_COUNT,
            "layers": build_manifest["layers"],
        },
        "prompt": {
            "messages": messages,
            "serialization": prompt,
            "token_ids": prompt_ids,
        },
        "capacity_plan": plan,
        "numeric_profile": {
            "projection": "native asymmetric packed INT4 AWQ W4A16 G128",
            "qzero_adjustment": "none",
            "activations": "FP16",
            "kv": "FP16",
            "independent_reference": (
                "PyTorch CPU float64 dequantized-AWQ Qwen2 with causal KV"
            ),
        },
        "rtl_execution": {
            "represented_positions": len(positions),
            "expected_layer_transactions": len(positions) * LAYER_COUNT,
            "natural_layer_transactions": sum(
                int(layer["natural_terminal"])
                for position in positions
                for layer in position["layers"]
            ),
            "positions": positions,
            "tensor_manifests": [
                tensor_manifests[layer_index]
                for layer_index in range(LAYER_COUNT)
            ],
        },
        "generation": {
            "max_new_tokens": max_new_tokens,
            "stop_reason": stop_reason,
            "eos_token_id": EOS_TOKEN_ID,
            "generated_token_ids": generated_ids,
            "generated_tokens": generated,
            "decoded_text": tokenizer.decode(
                decoded_ids,
                skip_special_tokens=False,
            ),
            "decisions": decisions,
        },
        "token0_behavior": {
            "definition": (
                "first generated token selected from the layer-23 output of "
                "the final prompt token"
            ),
            "selection_is_rtl_input": False,
            "selected_after_position": generated[0]["selected_after_position"],
            "fed_back_to_rtl": generated[0]["fed_back_to_rtl"],
            "rtl_input_position": generated[0]["rtl_input_position"],
            "final_selection_is_not_automatically_represented": True,
        },
        "claim_boundary": {
            "demonstrated": (
                "transactional indexed Verilator RTL for every listed "
                "represented token and all 24 layers, with authenticated "
                "persistent causal state and host finalization"
            ),
            "software_only_model24_relabelled_as_rtl": False,
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
        },
    }
    require(
        document["rtl_execution"]["natural_layer_transactions"]
        == document["rtl_execution"]["expected_layer_transactions"],
        "rtl_transaction_failed",
        "not every represented token traversed every indexed RTL layer",
    )
    write_json(output_dir / "execution.json", document)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "kind": "ace3_model24_first_voice_hybrid_manifest",
            "execution": hash_file(output_dir / "execution.json"),
            "represented_positions": len(positions),
            "layer_transactions": len(positions) * LAYER_COUNT,
            "generated_token_ids": generated_ids,
            "stop_reason": stop_reason,
            "token0_fed_back_to_rtl": generated[0]["fed_back_to_rtl"],
        },
    )
    return document


def blocker_document(error: HybridRtlError) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ace3_model24_first_voice_hybrid_blocker",
        "status": "blocked",
        "code": error.code,
        "message": str(error),
        "details": error.details,
        "rtl_claim": False,
        "software_fallback_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--compiled-dir", required=True, type=Path)
    parser.add_argument("--bind-compiled", action="store_true")
    parser.add_argument("--build-compact-layer", action="store_true")
    parser.add_argument("--layer-index", type=int)
    parser.add_argument("--temporary-mdir", type=Path)
    parser.add_argument("--verilator", default="verilator")
    parser.add_argument("--strip", default="strip")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--tensor-map", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    compiled_dir = args.compiled_dir.resolve()
    if args.build_compact_layer:
        if args.layer_index is None or args.temporary_mdir is None:
            raise SystemExit(
                "layer-index and temporary-mdir are required for compact build"
            )
        try:
            document = build_compact_layer(
                repository_root,
                compiled_dir,
                args.layer_index,
                args.temporary_mdir,
                verilator=args.verilator,
                strip=args.strip,
            )
        except HybridRtlError as error:
            raise SystemExit(
                f"MODEL24_FIRST_VOICE_COMPACT_BUILD_BLOCKED code={error.code} {error}"
            ) from error
        print(
            "MODEL24_FIRST_VOICE_COMPACT_BUILD_PASS "
            f"layer={document['layer_index']} "
            f"binary_sha256={document['binary']['sha256']}"
        )
        return
    compiled_dir = compiled_dir.resolve(strict=True)
    if args.bind_compiled:
        document = bind_compiled(repository_root, compiled_dir)
        print(
            "MODEL24_FIRST_VOICE_BUILD_BINDING_PASS "
            f"layers={len(document['layers'])} savable=1"
        )
        return
    if not all(
        value is not None
        for value in (
            args.checkpoint,
            args.tokenizer_dir,
            args.tensor_map,
            args.output_dir,
        )
    ):
        raise SystemExit(
            "checkpoint, tokenizer-dir, tensor-map, and output-dir are required"
        )
    output_dir = args.output_dir.resolve()
    try:
        document = execute(
            repository_root=repository_root,
            checkpoint_path=args.checkpoint.resolve(strict=True),
            tokenizer_dir=args.tokenizer_dir.resolve(strict=True),
            tensor_map_path=args.tensor_map.resolve(strict=True),
            compiled_dir=compiled_dir,
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
        )
    except HybridRtlError as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "blocker.json", blocker_document(error))
        raise SystemExit(
            f"MODEL24_FIRST_VOICE_HYBRID_BLOCKED code={error.code} {error}"
        ) from error
    print(
        "MODEL24_FIRST_VOICE_HYBRID_PASS "
        f"positions={document['rtl_execution']['represented_positions']} "
        f"transactions={document['rtl_execution']['natural_layer_transactions']} "
        f"token_ids={document['generation']['generated_token_ids']} "
        f"token0_feedback={int(document['token0_behavior']['fed_back_to_rtl'])}"
    )


if __name__ == "__main__":
    main()
