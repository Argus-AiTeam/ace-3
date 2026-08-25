#!/usr/bin/env python3
"""Authenticated numerical execution of all 24 official Qwen decoder layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as torch_functional
from safetensors import safe_open

from attention_oracle import attention_score, attention_softmax, attention_value
from fp16_adaptation_oracle import rmsnorm
from model24_execution_oracle import (
    DEFAULT_OFFICIAL_CHECKPOINT,
    DEFAULT_OFFICIAL_TOKENIZER_DIR,
    Model24TokenDecisionHost,
    authenticate_tokenizer,
    exact_tied_lm_head_logits,
)
from model24_oracle import (
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    OFFICIAL_CONFIG,
    authenticate_checkpoint,
    expected_tensor_specs,
)
from official_single_decoder_layer import (
    EMBED_TENSOR_SHA256,
    GROUP_SIZE,
    HEAD_DIM,
    HIDDEN_SIZE,
    KEY_VALUE_HEADS,
    LayerExecutionError,
    QUERY_HEADS,
    ROPE_THETA,
    TOKEN_IDS,
    TOKEN_ROW_SHA256,
    TOKEN_TEXT,
    _bits_to_f16,
    _canonical_bytes,
    _finite_rmsnorm,
    _f16_to_bits,
    _projection,
    _residual,
    _rope,
    _sha256,
    _torch_linear,
    _torch_rmsnorm,
)

MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
LAYER_COUNT = 24
LAYER_TENSOR_COUNT = 26
LAYER_STAGE_COUNT = 20
TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE = 0.125
LOGITS_ABSOLUTE_TOLERANCE = 0.125
ARTIFACT_NAME = "official_model24_next_token.json"
MANIFEST_NAME = "manifest.json"


class Model24ExecutionError(RuntimeError):
    """Raised when official Model24 execution or evidence validation fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Model24ExecutionError(message)


def _canonical_json(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_without_duplicates(payload: bytes, source: str) -> dict[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"{source} has duplicate key {key}")
            result[key] = value
        return result

    try:
        document = json.loads(payload, object_pairs_hook=object_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Model24ExecutionError(f"{source} is not valid JSON: {error}") from error
    _require(isinstance(document, dict), f"{source} root must be an object")
    return document


def _layer_tensor_names(layer_id: int) -> tuple[str, ...]:
    prefix = f"model.layers.{layer_id}."
    names = tuple(
        sorted(name for name in expected_tensor_specs() if name.startswith(prefix))
    )
    _require(
        len(names) == LAYER_TENSOR_COUNT,
        f"layer {layer_id} tensor inventory is not {LAYER_TENSOR_COUNT}",
    )
    return names


def _tensor_record(name: str, value: np.ndarray) -> dict[str, Any]:
    payload = _canonical_bytes(value)
    return {
        "name": name,
        "dtype": "F16" if value.dtype == np.float16 else "I32",
        "shape": list(value.shape),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _stage_record(name: str, bits: np.ndarray) -> dict[str, Any]:
    values = _bits_to_f16(bits).astype(np.float64)
    payload = _canonical_bytes(np.asarray(bits, dtype="<u2"))
    return {
        "name": name,
        "shape": list(bits.shape),
        "dtype": "F16",
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
        "finite": bool(np.all(np.isfinite(values))),
        "nonzero_count": int(np.count_nonzero(bits & 0x7FFF)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _stack_projection(
    activations: np.ndarray,
    tensors: dict[str, np.ndarray],
    prefix: str,
    bias_name: str | None = None,
) -> tuple[np.ndarray, int]:
    rows = []
    sampled_checks = 0
    for activation in activations:
        output, checks = _projection(activation, tensors, prefix, bias_name)
        rows.append(output)
        sampled_checks += checks
    return np.stack(rows), sampled_checks


def _fp16_silu_gate(gate_bits: int, up_bits: int) -> tuple[int, bool, bool]:
    values = _bits_to_f16(np.asarray([gate_bits, up_bits], dtype="<u2")).astype(
        np.float64
    )
    gate, up = (float(value) for value in values)
    if not math.isfinite(gate) or not math.isfinite(up):
        return 0, True, False
    if gate >= 0.0:
        sigmoid = 1.0 / (1.0 + math.exp(-gate))
    else:
        exponential = math.exp(gate)
        sigmoid = exponential / (1.0 + exponential)
    result = gate * sigmoid * up
    maximum = float(np.finfo(np.float16).max)
    saturated = abs(result) > maximum
    rounded = np.float16(math.copysign(maximum, result) if saturated else result)
    bits = int(np.asarray(rounded, dtype="<f2").view("<u2"))
    return bits, False, saturated


def _primary_layer(
    layer_id: int,
    tensors: dict[str, np.ndarray],
    hidden: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    prefix = f"model.layers.{layer_id}"
    stages: dict[str, np.ndarray] = {"input_hidden": hidden.copy()}
    norm1 = np.stack(
        [
            _finite_rmsnorm(row, tensors[f"{prefix}.input_layernorm.weight"])
            for row in hidden
        ]
    )
    stages["input_rmsnorm"] = norm1
    q, q_checks = _stack_projection(
        norm1,
        tensors,
        f"{prefix}.self_attn.q_proj",
        f"{prefix}.self_attn.q_proj.bias",
    )
    k, k_checks = _stack_projection(
        norm1,
        tensors,
        f"{prefix}.self_attn.k_proj",
        f"{prefix}.self_attn.k_proj.bias",
    )
    v, v_checks = _stack_projection(
        norm1,
        tensors,
        f"{prefix}.self_attn.v_proj",
        f"{prefix}.self_attn.v_proj.bias",
    )
    stages["q_proj"], stages["k_proj"], stages["v_proj"] = q, k, v
    rotated_q = np.stack(
        [_rope(row, QUERY_HEADS, position) for position, row in enumerate(q)]
    ).reshape(len(TOKEN_IDS), QUERY_HEADS, HEAD_DIM)
    rotated_k = np.stack(
        [_rope(row, KEY_VALUE_HEADS, position) for position, row in enumerate(k)]
    ).reshape(len(TOKEN_IDS), KEY_VALUE_HEADS, HEAD_DIM)
    values = v.reshape(len(TOKEN_IDS), KEY_VALUE_HEADS, HEAD_DIM)
    stages["q_rope"], stages["k_rope"] = rotated_q, rotated_k
    stages["kv_cache_k"], stages["kv_cache_v"] = rotated_k.copy(), values.copy()

    scores = np.zeros((len(TOKEN_IDS), QUERY_HEADS, len(TOKEN_IDS)), dtype="<u2")
    probabilities = np.zeros_like(scores)
    attended = np.zeros(
        (len(TOKEN_IDS), QUERY_HEADS, HEAD_DIM),
        dtype="<u2",
    )
    for position in range(len(TOKEN_IDS)):
        for query_head in range(QUERY_HEADS):
            kv_head = query_head // (QUERY_HEADS // KEY_VALUE_HEADS)
            score_row = []
            for key_position in range(position + 1):
                result = attention_score(
                    rotated_q[position, query_head].tolist(),
                    rotated_k[key_position, kv_head].tolist(),
                    [True] * HEAD_DIM,
                    position,
                    key_position,
                )
                _require(
                    not result.invalid and not result.cache_miss,
                    f"layer {layer_id} attention score failed",
                )
                score_row.append(result.score_f16)
                scores[position, query_head, key_position] = result.score_f16
            softmax = attention_softmax(
                score_row,
                list(range(position + 1)),
                [True] * (position + 1),
                [False] * (position + 1),
                [False] * (position + 1),
                position,
            )
            _require(
                not softmax.invalid
                and not softmax.cache_miss
                and not softmax.row_error,
                f"layer {layer_id} attention softmax failed",
            )
            for key_position, probability in enumerate(softmax.probabilities_f16):
                probabilities[position, query_head, key_position] = probability
            for dimension in range(HEAD_DIM):
                result = attention_value(
                    list(softmax.probabilities_f16),
                    [
                        int(values[key_position, kv_head, dimension])
                        for key_position in range(position + 1)
                    ],
                    [True] * (position + 1),
                    [False] * (position + 1),
                )
                _require(
                    not result.invalid
                    and not result.cache_miss
                    and not result.row_error,
                    f"layer {layer_id} attention value failed",
                )
                attended[position, query_head, dimension] = result.value_f16
    stages["attention_scores"] = scores
    stages["attention_probabilities"] = probabilities
    stages["attention_value"] = attended
    output, o_checks = _stack_projection(
        attended.reshape(len(TOKEN_IDS), HIDDEN_SIZE),
        tensors,
        f"{prefix}.self_attn.o_proj",
    )
    stages["o_proj"] = output
    residual1 = np.stack(
        [_residual(output[index], hidden[index]) for index in range(len(TOKEN_IDS))]
    )
    stages["attention_residual"] = residual1
    norm2 = np.stack(
        [
            _finite_rmsnorm(
                row,
                tensors[f"{prefix}.post_attention_layernorm.weight"],
            )
            for row in residual1
        ]
    )
    stages["post_attention_rmsnorm"] = norm2
    gate, gate_checks = _stack_projection(
        norm2,
        tensors,
        f"{prefix}.mlp.gate_proj",
    )
    up, up_checks = _stack_projection(
        norm2,
        tensors,
        f"{prefix}.mlp.up_proj",
    )
    stages["gate_proj"], stages["up_proj"] = gate, up
    activated_rows = []
    for gate_row, up_row in zip(gate, up, strict=True):
        activated = []
        for gate_item, up_item in zip(gate_row, up_row, strict=True):
            value, invalid, saturated = _fp16_silu_gate(
                int(gate_item), int(up_item)
            )
            _require(
                not invalid and not saturated,
                f"layer {layer_id} SiLU gate failed",
            )
            activated.append(value)
        activated_rows.append(np.asarray(activated, dtype="<u2"))
    activated = np.stack(activated_rows)
    stages["silu"] = activated
    down, down_checks = _stack_projection(
        activated,
        tensors,
        f"{prefix}.mlp.down_proj",
    )
    stages["down_proj"] = down
    post_layer = np.stack(
        [_residual(down[index], residual1[index]) for index in range(len(TOKEN_IDS))]
    )
    stages["post_layer_hidden"] = post_layer
    records = [_stage_record(name, value) for name, value in stages.items()]
    _require(len(records) == LAYER_STAGE_COUNT, "primary layer stage count mismatch")
    _require(
        all(record["finite"] and record["nonzero_count"] > 0 for record in records),
        f"layer {layer_id} produced a vacuous or nonfinite stage",
    )
    second_probabilities = _bits_to_f16(probabilities[1])
    _require(
        bool(np.all(second_probabilities[:, :2] > 0)),
        f"layer {layer_id} second token did not read both KV positions",
    )
    checks = (
        q_checks
        + k_checks
        + v_checks
        + o_checks
        + gate_checks
        + up_checks
        + down_checks
    )
    return post_layer, records, checks


def _reference_layer(
    layer_id: int,
    tensors: dict[str, np.ndarray],
    hidden: torch.Tensor,
) -> torch.Tensor:
    prefix = f"model.layers.{layer_id}"
    norm1 = _torch_rmsnorm(hidden, tensors[f"{prefix}.input_layernorm.weight"])
    q = _torch_linear(
        norm1,
        tensors,
        f"{prefix}.self_attn.q_proj",
        f"{prefix}.self_attn.q_proj.bias",
    )
    k = _torch_linear(
        norm1,
        tensors,
        f"{prefix}.self_attn.k_proj",
        f"{prefix}.self_attn.k_proj.bias",
    )
    v = _torch_linear(
        norm1,
        tensors,
        f"{prefix}.self_attn.v_proj",
        f"{prefix}.self_attn.v_proj.bias",
    )
    positions = torch.arange(len(TOKEN_IDS), dtype=torch.float64)
    frequencies = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
    )
    angles = torch.outer(positions, frequencies)
    cosine, sine = torch.cos(angles), torch.sin(angles)

    def rotate(values: torch.Tensor, heads: int) -> torch.Tensor:
        shaped = values.reshape(len(TOKEN_IDS), heads, HEAD_DIM)
        low = shaped[..., : HEAD_DIM // 2]
        high = shaped[..., HEAD_DIM // 2 :]
        return torch.cat(
            (
                low * cosine[:, None, :] - high * sine[:, None, :],
                high * cosine[:, None, :] + low * sine[:, None, :],
            ),
            dim=-1,
        )

    rotated_q = rotate(q, QUERY_HEADS)
    rotated_k = rotate(k, KEY_VALUE_HEADS)
    values = v.reshape(len(TOKEN_IDS), KEY_VALUE_HEADS, HEAD_DIM)
    attended = torch.zeros(
        (len(TOKEN_IDS), QUERY_HEADS, HEAD_DIM),
        dtype=torch.float64,
    )
    for position in range(len(TOKEN_IDS)):
        for query_head in range(QUERY_HEADS):
            kv_head = query_head // (QUERY_HEADS // KEY_VALUE_HEADS)
            score = (
                rotated_q[position, query_head]
                @ rotated_k[: position + 1, kv_head].T
                / (HEAD_DIM**0.5)
            )
            attended[position, query_head] = (
                torch.softmax(score, dim=-1) @ values[: position + 1, kv_head]
            )
    output = _torch_linear(
        attended.reshape(len(TOKEN_IDS), HIDDEN_SIZE),
        tensors,
        f"{prefix}.self_attn.o_proj",
    )
    residual1 = hidden + output
    norm2 = _torch_rmsnorm(
        residual1,
        tensors[f"{prefix}.post_attention_layernorm.weight"],
    )
    gate = _torch_linear(norm2, tensors, f"{prefix}.mlp.gate_proj")
    up = _torch_linear(norm2, tensors, f"{prefix}.mlp.up_proj")
    down = _torch_linear(
        torch_functional.silu(gate) * up,
        tensors,
        f"{prefix}.mlp.down_proj",
    )
    return residual1 + down


def _load_embeddings(checkpoint: Any) -> tuple[np.ndarray, dict[str, Any]]:
    embedding_slice = checkpoint.get_slice("model.embed_tokens.weight")
    embeddings = np.stack(
        [np.asarray(embedding_slice[token_id], dtype="<f2") for token_id in TOKEN_IDS]
    )
    for token_id, row in zip(TOKEN_IDS, embeddings, strict=True):
        _require(
            _sha256(row) == TOKEN_ROW_SHA256[token_id],
            f"embedding row {token_id} SHA256 mismatch",
        )
    return embeddings, {
        "name": "model.embed_tokens.weight",
        "dtype": "F16",
        "shape": [OFFICIAL_CONFIG["vocab_size"], HIDDEN_SIZE],
        "sha256": EMBED_TENSOR_SHA256,
        "consumed_rows": [
            {
                "token_id": token_id,
                "token": token,
                "sha256": TOKEN_ROW_SHA256[token_id],
            }
            for token_id, token in zip(TOKEN_IDS, TOKEN_TEXT, strict=True)
        ],
    }


def _reference_logits(
    checkpoint: Any,
    normalized_hidden: torch.Tensor,
    rows_per_chunk: int = 2048,
) -> np.ndarray:
    weight_slice = checkpoint.get_slice("lm_head.weight")
    chunks = []
    for start in range(0, OFFICIAL_CONFIG["vocab_size"], rows_per_chunk):
        end = min(start + rows_per_chunk, OFFICIAL_CONFIG["vocab_size"])
        weights = torch.from_numpy(
            np.asarray(weight_slice[start:end], dtype=np.float16).astype(np.float64)
        )
        chunks.append((weights @ normalized_hidden).detach().cpu().numpy())
    return np.concatenate(chunks)


def execute_model24(
    checkpoint_path: Path,
    tokenizer_dir: Path,
) -> dict[str, Any]:
    try:
        authenticate_checkpoint(checkpoint_path)
    except Exception as error:
        raise Model24ExecutionError(
            f"official checkpoint authentication failed: {error}"
        ) from error
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    torch.set_num_threads(1)
    layer_records = []
    sampled_checks = 0
    with safe_open(checkpoint_path, framework="np") as checkpoint:
        embeddings, embedding_record = _load_embeddings(checkpoint)
        primary_hidden = _f16_to_bits(embeddings)
        reference_hidden = torch.from_numpy(embeddings.astype(np.float64))
        for layer_id in range(LAYER_COUNT):
            names = _layer_tensor_names(layer_id)
            tensors = {name: np.asarray(checkpoint.get_tensor(name)) for name in names}
            tensor_records = [_tensor_record(name, tensors[name]) for name in names]
            input_hash = _sha256_bytes(_canonical_bytes(primary_hidden))
            primary_hidden, stage_records, checks = _primary_layer(
                layer_id,
                tensors,
                primary_hidden,
            )
            reference_hidden = _reference_layer(layer_id, tensors, reference_hidden)
            reference_values = reference_hidden.detach().cpu().numpy()
            primary_values = _bits_to_f16(primary_hidden).astype(np.float64)
            difference = np.abs(primary_values - reference_values)
            layer_records.append(
                {
                    "layer_id": layer_id,
                    "input_hidden_sha256": input_hash,
                    "consumed_tensors": tensor_records,
                    "intermediates": stage_records,
                    "output_hidden_sha256": _sha256_bytes(
                        _canonical_bytes(primary_hidden)
                    ),
                    "kv_cache_lineage": {
                        "positions": list(range(len(TOKEN_IDS))),
                        "k_sha256": next(
                            item["sha256"]
                            for item in stage_records
                            if item["name"] == "kv_cache_k"
                        ),
                        "v_sha256": next(
                            item["sha256"]
                            for item in stage_records
                            if item["name"] == "kv_cache_v"
                        ),
                    },
                    "independent_reference": {
                        "implementation": "PyTorch CPU float64 dequantized AWQ",
                        "max_abs_error": float(difference.max()),
                        "mean_abs_error": float(difference.mean()),
                    },
                }
            )
            sampled_checks += checks
            del tensors
        final_norm_weight = np.asarray(checkpoint.get_tensor("model.norm.weight"))
        primary_terminal = primary_hidden[-1]
        norm_outputs, mean_q48, rms_q24 = rmsnorm(
            primary_terminal.tolist(),
            _f16_to_bits(final_norm_weight).tolist(),
        )
        _require(
            all(not invalid and not saturated for _, invalid, saturated in norm_outputs),
            "final RMSNorm produced invalid or saturated output",
        )
        primary_normalized = np.asarray(
            [bits for bits, _, _ in norm_outputs],
            dtype="<u2",
        )
        primary_logits = exact_tied_lm_head_logits(
            checkpoint_path,
            primary_normalized.tolist(),
        )
        primary_logit_values = _bits_to_f16(
            np.asarray(primary_logits, dtype="<u2")
        ).astype(np.float64)
        reference_terminal = reference_hidden[-1]
        reference_normalized = _torch_rmsnorm(
            reference_terminal.unsqueeze(0),
            final_norm_weight,
        )[0]
        reference_logits = _reference_logits(
            checkpoint,
            reference_normalized,
        )

    terminal_difference = np.abs(
        _bits_to_f16(primary_terminal).astype(np.float64)
        - reference_terminal.detach().cpu().numpy()
    )
    logits_difference = np.abs(primary_logit_values - reference_logits)
    primary_argmax = int(np.argmax(primary_logit_values))
    reference_argmax = int(np.argmax(reference_logits))
    terminal_max = float(terminal_difference.max())
    logits_max = float(logits_difference.max())
    _require(
        terminal_max <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
        f"terminal hidden tolerance exceeded: {terminal_max}",
    )
    _require(
        logits_max <= LOGITS_ABSOLUTE_TOLERANCE,
        f"logits tolerance exceeded: {logits_max}",
    )
    _require(
        primary_argmax == reference_argmax,
        f"argmax mismatch: primary={primary_argmax} reference={reference_argmax}",
    )
    token_decision = Model24TokenDecisionHost(tokenizer).decide(primary_logits)
    _require(
        token_decision["argmax_token_id"] == primary_argmax,
        "token-decision host argmax mismatch",
    )
    terminal_payload = _canonical_bytes(primary_terminal)
    normalized_payload = _canonical_bytes(primary_normalized)
    logits_payload = _canonical_bytes(np.asarray(primary_logits, dtype="<u2"))
    return {
        "schema_version": 1,
        "kind": "ace3_official_model24_next_token",
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint": {
                "filename": checkpoint_path.name,
                "sha256": CHECKPOINT_SHA256,
                "bytes": CHECKPOINT_SIZE,
            },
            "embedding": embedding_record,
            "final_norm": _tensor_record("model.norm.weight", final_norm_weight),
            "lm_head": {
                "name": "lm_head.weight",
                "dtype": "F16",
                "shape": [OFFICIAL_CONFIG["vocab_size"], HIDDEN_SIZE],
                "tied_to": "model.embed_tokens.weight",
                "sha256": EMBED_TENSOR_SHA256,
            },
        },
        "prompt": {
            "utf8": "".join(TOKEN_TEXT),
            "token_ids": list(TOKEN_IDS),
            "positions": list(range(len(TOKEN_IDS))),
            "decision_position": len(TOKEN_IDS) - 1,
        },
        "numeric_profile": {
            "projection": "native asymmetric packed INT4 AWQ W4A16 G128",
            "qzero_adjustment": "none",
            "activations": "FP16",
            "scales": "FP16",
            "kv_cache": "per-layer FP16 causal K/V",
            "rope": "Qwen half-split theta=1000000",
            "residual_rmsnorm": "accepted ACE-3 FP16 software primitives",
            "silu": "mathematical SiLU independently evaluated and rounded to FP16",
        },
        "layers": layer_records,
        "execution_summary": {
            "layer_count": len(layer_records),
            "consumed_layer_tensor_count": sum(
                len(layer["consumed_tensors"]) for layer in layer_records
            ),
            "intermediate_hash_count": sum(
                len(layer["intermediates"]) for layer in layer_records
            ),
            "sampled_projection_bit_oracle_checks": sampled_checks,
            "all_layers_causal_kv_positions": list(range(len(TOKEN_IDS))),
        },
        "terminal_hidden_state": {
            "source": "model.layers.23 post_layer_hidden[token_index=1]",
            "source_layer_output_sha256": layer_records[-1]["output_hidden_sha256"],
            "shape": [HIDDEN_SIZE],
            "dtype": "F16",
            "sha256": _sha256_bytes(terminal_payload),
            "f16_bits": [int(value) for value in primary_terminal],
            "independent_reference": {
                "implementation": "PyTorch CPU float64 dequantized-AWQ Qwen2",
                "max_abs_error": terminal_max,
                "mean_abs_error": float(terminal_difference.mean()),
                "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
                "within_tolerance": True,
            },
        },
        "final_rmsnorm": {
            "input_sha256": _sha256_bytes(terminal_payload),
            "output_sha256": _sha256_bytes(normalized_payload),
            "mean_q48": mean_q48,
            "rms_q24": rms_q24,
        },
        "lm_head": {
            "input_sha256": _sha256_bytes(normalized_payload),
            "logits_sha256": _sha256_bytes(logits_payload),
            "vocab_size": len(primary_logits),
            "independent_reference": {
                "implementation": "PyTorch CPU float64 chunked tied lm_head",
                "max_abs_error": logits_max,
                "mean_abs_error": float(logits_difference.mean()),
                "absolute_tolerance": LOGITS_ABSOLUTE_TOLERANCE,
                "within_tolerance": True,
            },
        },
        "token_decision": {
            **token_decision,
            "independent_reference_argmax_token_id": reference_argmax,
            "argmax_matches_independent_reference": True,
        },
        "claim_boundary": {
            "demonstrated": (
                "one deterministic next-token decision after authenticated numerical "
                "execution of two fixed tokens through all 24 software/oracle layers"
            ),
            "multi_token_dialogue": "not demonstrated",
            "rtl": "layers 1 through 23 and full-model integration not demonstrated",
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
            "latency": "not measured",
            "throughput": "not measured",
        },
    }


def validate_document(document: Mapping[str, Any]) -> dict[str, Any]:
    _require(document.get("schema_version") == 1, "evidence schema version mismatch")
    _require(
        document.get("kind") == "ace3_official_model24_next_token",
        "evidence kind mismatch",
    )
    checkpoint = document["model_binding"]["checkpoint"]
    _require(checkpoint["sha256"] == CHECKPOINT_SHA256, "checkpoint SHA256 binding mismatch")
    _require(checkpoint["bytes"] == CHECKPOINT_SIZE, "checkpoint byte binding mismatch")
    layers = document["layers"]
    _require(len(layers) == LAYER_COUNT, "evidence must contain 24 layers")
    previous_output = None
    all_tensor_names = set()
    for layer_id, layer in enumerate(layers):
        _require(layer["layer_id"] == layer_id, "layer ordering mismatch")
        if previous_output is not None:
            _require(
                layer["input_hidden_sha256"] == previous_output,
                f"layer {layer_id} hidden-state lineage mismatch",
            )
        previous_output = layer["output_hidden_sha256"]
        tensors = layer["consumed_tensors"]
        _require(
            len(tensors) == LAYER_TENSOR_COUNT,
            f"layer {layer_id} tensor count mismatch",
        )
        expected_names = set(_layer_tensor_names(layer_id))
        actual_names = {record["name"] for record in tensors}
        _require(actual_names == expected_names, f"layer {layer_id} tensor inventory mismatch")
        _require(
            not (all_tensor_names & actual_names),
            f"layer {layer_id} reuses another layer tensor",
        )
        all_tensor_names.update(actual_names)
        intermediates = layer["intermediates"]
        _require(
            len(intermediates) == LAYER_STAGE_COUNT,
            f"layer {layer_id} intermediate count mismatch",
        )
        intermediate_hashes = {record["name"]: record["sha256"] for record in intermediates}
        _require(
            layer["kv_cache_lineage"]["k_sha256"]
            == intermediate_hashes["kv_cache_k"],
            f"layer {layer_id} K-cache lineage mismatch",
        )
        _require(
            layer["kv_cache_lineage"]["v_sha256"]
            == intermediate_hashes["kv_cache_v"],
            f"layer {layer_id} V-cache lineage mismatch",
        )
        _require(
            layer["output_hidden_sha256"]
            == intermediate_hashes["post_layer_hidden"],
            f"layer {layer_id} output hash mismatch",
        )
    summary = document["execution_summary"]
    _require(
        summary["consumed_layer_tensor_count"] == LAYER_COUNT * LAYER_TENSOR_COUNT,
        "consumed layer tensor total mismatch",
    )
    _require(
        summary["intermediate_hash_count"] == LAYER_COUNT * LAYER_STAGE_COUNT,
        "intermediate hash total mismatch",
    )
    terminal = document["terminal_hidden_state"]
    _require(
        terminal["source_layer_output_sha256"] == previous_output,
        "terminal hidden does not descend from layer 23",
    )
    terminal_bits = np.asarray(terminal["f16_bits"], dtype="<u2")
    _require(
        tuple(terminal_bits.shape) == (HIDDEN_SIZE,)
        and terminal["sha256"]
        == _sha256_bytes(_canonical_bytes(terminal_bits)),
        "terminal hidden row hash mismatch",
    )
    terminal_comparison = terminal["independent_reference"]
    _require(
        terminal_comparison["absolute_tolerance"]
        == TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
        "terminal hidden tolerance mismatch",
    )
    _require(
        terminal_comparison["within_tolerance"]
        and terminal_comparison["max_abs_error"]
        <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
        "terminal hidden comparison failed",
    )
    logits_comparison = document["lm_head"]["independent_reference"]
    _require(
        logits_comparison["absolute_tolerance"] == LOGITS_ABSOLUTE_TOLERANCE,
        "logits tolerance mismatch",
    )
    _require(
        logits_comparison["within_tolerance"]
        and logits_comparison["max_abs_error"] <= LOGITS_ABSOLUTE_TOLERANCE,
        "logits comparison failed",
    )
    _require(
        document["token_decision"]["argmax_matches_independent_reference"],
        "argmax does not match independent reference",
    )
    _require(
        document["token_decision"]["argmax_token_id"]
        == document["token_decision"]["independent_reference_argmax_token_id"],
        "argmax token IDs differ",
    )
    claims = document["claim_boundary"]
    for name in (
        "multi_token_dialogue",
        "rtl",
        "synthesis",
        "ppa",
        "fpga",
        "latency",
        "throughput",
    ):
        _require(name in claims, f"missing {name} claim boundary")
    return {
        "layers": len(layers),
        "layer_tensors": len(all_tensor_names),
        "intermediate_hashes": summary["intermediate_hash_count"],
        "argmax_token_id": document["token_decision"]["argmax_token_id"],
        "decoded_token": document["token_decision"]["argmax_decoded_token"],
    }


def authenticate_evidence_tensors(
    document: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    try:
        authenticate_checkpoint(checkpoint_path)
    except Exception as error:
        raise Model24ExecutionError(
            f"official checkpoint authentication failed: {error}"
        ) from error
    records = {
        record["name"]: record
        for layer in document["layers"]
        for record in layer["consumed_tensors"]
    }
    with safe_open(checkpoint_path, framework="np") as checkpoint:
        for name in sorted(records):
            value = np.asarray(checkpoint.get_tensor(name))
            actual = _tensor_record(name, value)
            _require(actual == records[name], f"{name} authenticated tensor record mismatch")


def generate(
    output_dir: Path,
    checkpoint_path: Path = DEFAULT_OFFICIAL_CHECKPOINT,
    tokenizer_dir: Path = DEFAULT_OFFICIAL_TOKENIZER_DIR,
) -> dict[str, bytes]:
    document = execute_model24(checkpoint_path, tokenizer_dir)
    validate_document(document)
    evidence_payload = _canonical_json(document)
    manifest = {
        "schema_version": 1,
        "kind": "ace3_official_model24_next_token_manifest",
        "artifacts": {
            ARTIFACT_NAME: {
                "bytes": len(evidence_payload),
                "sha256": _sha256_bytes(evidence_payload),
            }
        },
        "summary": validate_document(document),
    }
    payloads = {
        ARTIFACT_NAME: evidence_payload,
        MANIFEST_NAME: _canonical_json(manifest),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.iterdir()}
    _require(
        existing <= set(payloads),
        f"output directory contains unexpected files: {sorted(existing - set(payloads))}",
    )
    for name, payload in payloads.items():
        (output_dir / name).write_bytes(payload)
    return payloads


def validate_directory(
    vector_dir: Path,
    checkpoint_path: Path = DEFAULT_OFFICIAL_CHECKPOINT,
) -> dict[str, Any]:
    _require(vector_dir.is_dir(), f"evidence directory is missing: {vector_dir}")
    actual_names = {path.name for path in vector_dir.iterdir() if path.is_file()}
    _require(
        actual_names == {ARTIFACT_NAME, MANIFEST_NAME},
        "evidence artifact set mismatch",
    )
    evidence_payload = (vector_dir / ARTIFACT_NAME).read_bytes()
    manifest = _json_without_duplicates(
        (vector_dir / MANIFEST_NAME).read_bytes(),
        MANIFEST_NAME,
    )
    _require(
        manifest["kind"] == "ace3_official_model24_next_token_manifest",
        "manifest kind mismatch",
    )
    record = manifest["artifacts"][ARTIFACT_NAME]
    _require(record["bytes"] == len(evidence_payload), "evidence byte count mismatch")
    _require(
        record["sha256"] == _sha256_bytes(evidence_payload),
        "evidence SHA256 mismatch",
    )
    document = _json_without_duplicates(evidence_payload, ARTIFACT_NAME)
    summary = validate_document(document)
    _require(summary == manifest["summary"], "manifest summary mismatch")
    authenticate_evidence_tensors(document, checkpoint_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("generate", "validate"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--vector-dir", type=Path)
    parser.add_argument(
        "--official-checkpoint",
        type=Path,
        default=DEFAULT_OFFICIAL_CHECKPOINT,
    )
    parser.add_argument(
        "--official-tokenizer-dir",
        type=Path,
        default=DEFAULT_OFFICIAL_TOKENIZER_DIR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.operation == "generate":
            _require(args.output_dir is not None, "--output-dir is required")
            payloads = generate(
                args.output_dir.resolve(),
                args.official_checkpoint.resolve(),
                args.official_tokenizer_dir.resolve(),
            )
            evidence = _json_without_duplicates(
                payloads[ARTIFACT_NAME],
                ARTIFACT_NAME,
            )
            summary = validate_document(evidence)
            print(
                "OFFICIAL_MODEL24_NEXT_TOKEN_GENERATION_PASS "
                f"layers={summary['layers']} "
                f"layer_tensors={summary['layer_tensors']} "
                f"intermediate_hashes={summary['intermediate_hashes']} "
                f"argmax_token_id={summary['argmax_token_id']} "
                f"decoded_token={summary['decoded_token']!r}"
            )
        else:
            _require(args.vector_dir is not None, "--vector-dir is required")
            summary = validate_directory(
                args.vector_dir.resolve(),
                args.official_checkpoint.resolve(),
            )
            print(
                "OFFICIAL_MODEL24_NEXT_TOKEN_VALIDATION_PASS "
                f"layers={summary['layers']} "
                f"layer_tensors={summary['layer_tensors']} "
                f"intermediate_hashes={summary['intermediate_hashes']} "
                f"argmax_token_id={summary['argmax_token_id']} "
                f"decoded_token={summary['decoded_token']!r}"
            )
    except (LayerExecutionError, Model24ExecutionError, OSError) as error:
        raise SystemExit(f"OFFICIAL_MODEL24_NEXT_TOKEN_FAIL {error}") from error


if __name__ == "__main__":
    main()
