#!/usr/bin/env python3
"""Deterministic official-checkpoint dialogue generation with FP16 KV lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional
from safetensors import safe_open

from attention_oracle import attention_score, attention_softmax, attention_value
from fp16_adaptation_oracle import rmsnorm
from model24_execution_oracle import (
    DEFAULT_OFFICIAL_CHECKPOINT,
    DEFAULT_OFFICIAL_TOKENIZER_DIR,
    EOS_TOKEN_ID,
    FIXED_CHAT_MESSAGES,
    FIXED_CHAT_SERIALIZATION,
    FIXED_CHAT_TOKEN_IDS,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_SHA256,
    Model24TokenDecisionHost,
    _decode_f16_array_q24,
    authenticate_tokenizer,
    exact_tied_lm_head_logits,
    serialize_chat_prompt,
)
from model24_oracle import (
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    OFFICIAL_CONFIG,
    authenticate_checkpoint,
)
from official_model24_next_token import (
    HIDDEN_SIZE,
    KEY_VALUE_HEADS,
    LAYER_COUNT,
    LAYER_TENSOR_COUNT,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    QUERY_HEADS,
    TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
    _canonical_bytes,
    _canonical_json,
    _finite_rmsnorm,
    _fp16_silu_gate,
    _json_without_duplicates,
    _layer_tensor_names,
    _reference_logits,
    _require,
    _residual,
    _rope,
    _sha256_bytes,
)
from official_single_decoder_layer import (
    GROUP_SIZE,
    HEAD_DIM,
    _bits_to_f16,
    _f16_to_bits,
    _torch_rmsnorm,
    _unpack_words,
)
from awq_bit_oracle import q47_48_to_f16

ARTIFACT_NAME = "official_model24_dialogue.json"
MANIFEST_NAME = "manifest.json"
EVIDENCE_KIND = "ace3_official_model24_multitoken_dialogue"
DEFAULT_MAX_NEW_TOKENS = 4
LOGITS_ABSOLUTE_TOLERANCE = 0.25
MODEL24_BINDING_RELATIVE_PATH = (
    "ace3/contracts/model24_execution_vector_bindings.json"
)
MODEL24_BINDING_SHA256 = (
    "79389eda61e1bf2b59cf93f834bb6705d38cede797aa07479fc029109c150df1"
)


@dataclass
class PreparedProjection:
    delta: np.ndarray
    scale_q24: np.ndarray
    bias_bits: np.ndarray | None
    reference_weight: torch.Tensor
    reference_bias: torch.Tensor | None


@dataclass
class LayerState:
    layer_id: int
    input_norm: np.ndarray
    post_attention_norm: np.ndarray
    projections: dict[str, PreparedProjection]
    primary_k: np.ndarray
    primary_v: np.ndarray
    reference_k: torch.Tensor
    reference_v: torch.Tensor


class DialogueExecutionError(RuntimeError):
    """Raised when dialogue execution or evidence validation fails."""


def _dialogue_require(condition: bool, message: str) -> None:
    if not condition:
        raise DialogueExecutionError(message)


def _binding_path() -> Path:
    return Path(__file__).resolve().parents[2] / MODEL24_BINDING_RELATIVE_PATH


def _authenticate_model24_binding(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    _dialogue_require(
        _sha256_bytes(payload) == MODEL24_BINDING_SHA256,
        "accepted Model24 vector binding SHA256 mismatch",
    )
    document = _json_without_duplicates(payload, path.name)
    _dialogue_require(
        document.get("kind") == "ace3_model24_execution_vector_bindings",
        "accepted Model24 vector binding kind mismatch",
    )
    inputs = document.get("inputs", {})
    _dialogue_require(
        inputs.get("checkpoint_sha256") == CHECKPOINT_SHA256,
        "accepted Model24 binding checkpoint mismatch",
    )
    _dialogue_require(
        inputs.get("tokenizer_sha256") == TOKENIZER_SHA256
        and inputs.get("tokenizer_config_sha256") == TOKENIZER_CONFIG_SHA256,
        "accepted Model24 binding tokenizer mismatch",
    )
    return {
        "path": MODEL24_BINDING_RELATIVE_PATH,
        "sha256": MODEL24_BINDING_SHA256,
        "kind": document["kind"],
        "checkpoint_sha256": inputs["checkpoint_sha256"],
        "tokenizer_sha256": inputs["tokenizer_sha256"],
        "tokenizer_config_sha256": inputs["tokenizer_config_sha256"],
    }


def _prepare_projection(
    tensors: Mapping[str, np.ndarray],
    prefix: str,
    bias_name: str | None = None,
) -> PreparedProjection:
    quantized = _unpack_words(tensors[f"{prefix}.qweight"])
    zeros = _unpack_words(tensors[f"{prefix}.qzeros"])
    scales = np.asarray(tensors[f"{prefix}.scales"], dtype="<f2")
    groups = quantized.shape[0] // GROUP_SIZE
    _dialogue_require(
        zeros.shape == scales.shape
        and zeros.shape[0] == groups
        and zeros.shape[1] == quantized.shape[1],
        f"{prefix} AWQ geometry mismatch",
    )
    delta = (
        quantized.reshape(groups, GROUP_SIZE, quantized.shape[1])
        - zeros[:, None, :]
    ).astype(np.int16)
    scale_q24 = _decode_f16_array_q24(_f16_to_bits(scales))
    reference_values = delta.reshape(quantized.shape).astype(np.float64)
    reference_values *= np.repeat(
        scales.astype(np.float64),
        GROUP_SIZE,
        axis=0,
    )
    bias = None if bias_name is None else np.asarray(tensors[bias_name], dtype="<f2")
    return PreparedProjection(
        delta=delta,
        scale_q24=scale_q24,
        bias_bits=None if bias is None else _f16_to_bits(bias),
        reference_weight=torch.from_numpy(reference_values),
        reference_bias=(
            None
            if bias is None
            else torch.from_numpy(bias.astype(np.float64))
        ),
    )


def _exact_projection(
    activation_bits: np.ndarray,
    projection: PreparedProjection,
) -> np.ndarray:
    rows = np.asarray(activation_bits, dtype="<u2")
    _dialogue_require(rows.ndim == 2, "projection activation must be rank two")
    activation_q24 = _decode_f16_array_q24(rows).reshape(
        rows.shape[0],
        projection.delta.shape[0],
        GROUP_SIZE,
    )
    dots = np.einsum(
        "ngi,gio->ngo",
        activation_q24,
        projection.delta,
        dtype=np.int64,
        optimize=True,
    )
    accumulators = np.sum(
        dots.astype(object) * projection.scale_q24[None, :, :].astype(object),
        axis=1,
    )
    flat = np.fromiter(
        (
            q47_48_to_f16(int(accumulator))[0]
            for accumulator in accumulators.reshape(-1)
        ),
        dtype="<u2",
        count=accumulators.size,
    )
    output = flat.reshape(accumulators.shape)
    if projection.bias_bits is not None:
        output = np.stack(
            [_residual(row, projection.bias_bits) for row in output]
        )
    return output


def _reference_projection(
    activation: torch.Tensor,
    projection: PreparedProjection,
) -> torch.Tensor:
    output = activation.to(torch.float64) @ projection.reference_weight
    if projection.reference_bias is not None:
        output = output + projection.reference_bias
    return output


def _empty_primary_cache() -> np.ndarray:
    return np.zeros((0, KEY_VALUE_HEADS, HEAD_DIM), dtype="<u2")


def _empty_reference_cache() -> torch.Tensor:
    return torch.zeros((0, KEY_VALUE_HEADS, HEAD_DIM), dtype=torch.float64)


def _load_model(
    checkpoint_path: Path,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, list[LayerState]]:
    layers = []
    with safe_open(checkpoint_path, framework="np") as checkpoint:
        embeddings = np.asarray(
            checkpoint.get_tensor("model.embed_tokens.weight"),
            dtype="<f2",
        )
        final_norm = np.asarray(checkpoint.get_tensor("model.norm.weight"), dtype="<f2")
        for layer_id in range(LAYER_COUNT):
            names = _layer_tensor_names(layer_id)
            tensors = {
                name: np.asarray(checkpoint.get_tensor(name))
                for name in names
            }
            prefix = f"model.layers.{layer_id}"
            projections = {
                "q": _prepare_projection(
                    tensors,
                    f"{prefix}.self_attn.q_proj",
                    f"{prefix}.self_attn.q_proj.bias",
                ),
                "k": _prepare_projection(
                    tensors,
                    f"{prefix}.self_attn.k_proj",
                    f"{prefix}.self_attn.k_proj.bias",
                ),
                "v": _prepare_projection(
                    tensors,
                    f"{prefix}.self_attn.v_proj",
                    f"{prefix}.self_attn.v_proj.bias",
                ),
                "o": _prepare_projection(tensors, f"{prefix}.self_attn.o_proj"),
                "gate": _prepare_projection(tensors, f"{prefix}.mlp.gate_proj"),
                "up": _prepare_projection(tensors, f"{prefix}.mlp.up_proj"),
                "down": _prepare_projection(tensors, f"{prefix}.mlp.down_proj"),
            }
            layers.append(
                LayerState(
                    layer_id=layer_id,
                    input_norm=np.asarray(
                        tensors[f"{prefix}.input_layernorm.weight"],
                        dtype="<f2",
                    ),
                    post_attention_norm=np.asarray(
                        tensors[f"{prefix}.post_attention_layernorm.weight"],
                        dtype="<f2",
                    ),
                    projections=projections,
                    primary_k=_empty_primary_cache(),
                    primary_v=_empty_primary_cache(),
                    reference_k=_empty_reference_cache(),
                    reference_v=_empty_reference_cache(),
                )
            )
    _dialogue_require(
        embeddings.shape == (OFFICIAL_CONFIG["vocab_size"], HIDDEN_SIZE),
        "embedding geometry mismatch",
    )
    return (
        embeddings,
        final_norm,
        torch.from_numpy(embeddings.astype(np.float64)),
        layers,
    )


def _reset_kv_caches(states: Sequence[LayerState]) -> None:
    for state in states:
        state.primary_k = _empty_primary_cache()
        state.primary_v = _empty_primary_cache()
        state.reference_k = _empty_reference_cache()
        state.reference_v = _empty_reference_cache()


def _primary_layer_step(
    state: LayerState,
    hidden: np.ndarray,
    start_position: int,
) -> np.ndarray:
    rows = hidden.shape[0]
    norm1 = np.stack(
        [_finite_rmsnorm(row, state.input_norm) for row in hidden]
    )
    q = _exact_projection(norm1, state.projections["q"])
    k = _exact_projection(norm1, state.projections["k"])
    v = _exact_projection(norm1, state.projections["v"])
    rotated_q = np.stack(
        [
            _rope(row, QUERY_HEADS, start_position + offset)
            for offset, row in enumerate(q)
        ]
    ).reshape(rows, QUERY_HEADS, HEAD_DIM)
    rotated_k = np.stack(
        [
            _rope(row, KEY_VALUE_HEADS, start_position + offset)
            for offset, row in enumerate(k)
        ]
    ).reshape(rows, KEY_VALUE_HEADS, HEAD_DIM)
    values = v.reshape(rows, KEY_VALUE_HEADS, HEAD_DIM)
    state.primary_k = np.concatenate((state.primary_k, rotated_k), axis=0)
    state.primary_v = np.concatenate((state.primary_v, values), axis=0)

    attended = np.zeros((rows, QUERY_HEADS, HEAD_DIM), dtype="<u2")
    for offset in range(rows):
        position = start_position + offset
        for query_head in range(QUERY_HEADS):
            kv_head = query_head // (QUERY_HEADS // KEY_VALUE_HEADS)
            scores = []
            for key_position in range(position + 1):
                score = attention_score(
                    rotated_q[offset, query_head].tolist(),
                    state.primary_k[key_position, kv_head].tolist(),
                    [True] * HEAD_DIM,
                    position,
                    key_position,
                )
                _dialogue_require(
                    not score.invalid and not score.cache_miss,
                    f"layer {state.layer_id} attention score failed",
                )
                scores.append(score.score_f16)
            probabilities = attention_softmax(
                scores,
                list(range(position + 1)),
                [True] * (position + 1),
                [False] * (position + 1),
                [False] * (position + 1),
                position,
            )
            _dialogue_require(
                not probabilities.invalid
                and not probabilities.cache_miss
                and not probabilities.row_error,
                f"layer {state.layer_id} attention softmax failed",
            )
            for dimension in range(HEAD_DIM):
                value = attention_value(
                    list(probabilities.probabilities_f16),
                    [
                        int(state.primary_v[key_position, kv_head, dimension])
                        for key_position in range(position + 1)
                    ],
                    [True] * (position + 1),
                    [False] * (position + 1),
                )
                _dialogue_require(
                    not value.invalid
                    and not value.cache_miss
                    and not value.row_error,
                    f"layer {state.layer_id} attention value failed",
                )
                attended[offset, query_head, dimension] = value.value_f16

    output = _exact_projection(
        attended.reshape(rows, HIDDEN_SIZE),
        state.projections["o"],
    )
    residual1 = np.stack(
        [_residual(output[index], hidden[index]) for index in range(rows)]
    )
    norm2 = np.stack(
        [_finite_rmsnorm(row, state.post_attention_norm) for row in residual1]
    )
    gate = _exact_projection(norm2, state.projections["gate"])
    up = _exact_projection(norm2, state.projections["up"])
    activated = np.empty_like(gate)
    for row in range(rows):
        for channel in range(gate.shape[1]):
            bits, invalid, saturated = _fp16_silu_gate(
                int(gate[row, channel]),
                int(up[row, channel]),
            )
            _dialogue_require(
                not invalid and not saturated,
                f"layer {state.layer_id} SiLU gate failed",
            )
            activated[row, channel] = bits
    down = _exact_projection(activated, state.projections["down"])
    return np.stack(
        [_residual(down[index], residual1[index]) for index in range(rows)]
    )


def _reference_layer_step(
    state: LayerState,
    hidden: torch.Tensor,
    start_position: int,
) -> torch.Tensor:
    rows = hidden.shape[0]
    norm1 = _torch_rmsnorm(hidden, state.input_norm)
    q = _reference_projection(norm1, state.projections["q"])
    k = _reference_projection(norm1, state.projections["k"])
    v = _reference_projection(norm1, state.projections["v"])
    positions = torch.arange(
        start_position,
        start_position + rows,
        dtype=torch.float64,
    )
    frequencies = 1.0 / (
        1_000_000.0
        ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
    )
    angles = torch.outer(positions, frequencies)
    cosine, sine = torch.cos(angles), torch.sin(angles)

    def rotate(values: torch.Tensor, heads: int) -> torch.Tensor:
        shaped = values.reshape(rows, heads, HEAD_DIM)
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
    values = v.reshape(rows, KEY_VALUE_HEADS, HEAD_DIM)
    state.reference_k = torch.cat((state.reference_k, rotated_k), dim=0)
    state.reference_v = torch.cat((state.reference_v, values), dim=0)
    kv_indices = torch.arange(QUERY_HEADS) // (
        QUERY_HEADS // KEY_VALUE_HEADS
    )
    expanded_k = state.reference_k[:, kv_indices, :]
    expanded_v = state.reference_v[:, kv_indices, :]
    scores = torch.einsum(
        "bhd,khd->bhk",
        rotated_q,
        expanded_k,
    ) / math.sqrt(HEAD_DIM)
    key_positions = torch.arange(state.reference_k.shape[0])
    causal = key_positions[None, :] <= positions.to(torch.int64)[:, None]
    scores = scores.masked_fill(~causal[:, None, :], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    attended = torch.einsum("bhk,khd->bhd", probabilities, expanded_v)
    output = _reference_projection(
        attended.reshape(rows, HIDDEN_SIZE),
        state.projections["o"],
    )
    residual1 = hidden + output
    norm2 = _torch_rmsnorm(residual1, state.post_attention_norm)
    gate = _reference_projection(norm2, state.projections["gate"])
    up = _reference_projection(norm2, state.projections["up"])
    down = _reference_projection(
        torch_functional.silu(gate) * up,
        state.projections["down"],
    )
    return residual1 + down


def _execute_positions(
    states: Sequence[LayerState],
    primary_hidden: np.ndarray,
    reference_hidden: torch.Tensor,
    start_position: int,
) -> tuple[np.ndarray, torch.Tensor]:
    for state in states:
        primary_hidden = _primary_layer_step(
            state,
            primary_hidden,
            start_position,
        )
        reference_hidden = _reference_layer_step(
            state,
            reference_hidden,
            start_position,
        )
    return primary_hidden, reference_hidden


def _cache_snapshot(
    states: Sequence[LayerState],
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    position_count = states[0].primary_k.shape[0]
    _dialogue_require(
        all(
            state.primary_k.shape[0] == position_count
            and state.primary_v.shape[0] == position_count
            for state in states
        ),
        "per-layer primary KV cache lengths differ",
    )
    parent_layers = (
        {}
        if previous is None
        else {
            layer["layer_id"]: layer
            for layer in previous["layers"]
        }
    )
    layer_records = []
    for state in states:
        parent = parent_layers.get(state.layer_id)
        layer_records.append(
            {
                "layer_id": state.layer_id,
                "parent_k_sha256": None if parent is None else parent["k_sha256"],
                "parent_v_sha256": None if parent is None else parent["v_sha256"],
                "k_sha256": _sha256_bytes(_canonical_bytes(state.primary_k)),
                "v_sha256": _sha256_bytes(_canonical_bytes(state.primary_v)),
                "appended_k_row_sha256": _sha256_bytes(
                    _canonical_bytes(state.primary_k[-1])
                ),
                "appended_v_row_sha256": _sha256_bytes(
                    _canonical_bytes(state.primary_v[-1])
                ),
            }
        )
    aggregate = _sha256_bytes(
        _canonical_json(
            {
                "position_count": position_count,
                "layers": [
                    {
                        "layer_id": layer["layer_id"],
                        "k_sha256": layer["k_sha256"],
                        "v_sha256": layer["v_sha256"],
                    }
                    for layer in layer_records
                ],
            }
        )
    )
    parent_count = 0 if previous is None else previous["position_count"]
    return {
        "parent_cache_sha256": (
            None if previous is None else previous["cache_sha256"]
        ),
        "cache_sha256": aggregate,
        "parent_position_count": parent_count,
        "position_count": position_count,
        "added_positions": list(range(parent_count, position_count)),
        "layers": layer_records,
    }


def generation_stop_reason(
    token_id: int,
    generated_count: int,
    max_new_tokens: int,
) -> str | None:
    _dialogue_require(generated_count > 0, "generated count must be positive")
    _dialogue_require(
        generated_count <= max_new_tokens,
        "generated count exceeds maximum",
    )
    if token_id == EOS_TOKEN_ID:
        return "eos_token"
    if generated_count == max_new_tokens:
        return "max_new_tokens"
    return None


def _decision_record(
    checkpoint: Any,
    checkpoint_path: Path,
    tokenizer: Any,
    final_norm: np.ndarray,
    reference_lm_head: torch.Tensor,
    primary_terminal: np.ndarray,
    reference_terminal: torch.Tensor,
    cache: Mapping[str, Any],
    ordinal: int,
    generated_ids: Sequence[int],
    *,
    enforce_tolerances: bool = True,
) -> dict[str, Any]:
    norm_outputs, _, _ = rmsnorm(
        primary_terminal.tolist(),
        _f16_to_bits(final_norm).tolist(),
    )
    _dialogue_require(
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
    primary_values = _bits_to_f16(
        np.asarray(primary_logits, dtype="<u2")
    ).astype(np.float64)
    reference_normalized = _torch_rmsnorm(
        reference_terminal.unsqueeze(0),
        final_norm,
    )[0]
    reference_logits = (
        reference_lm_head @ reference_normalized
    ).detach().cpu().numpy()
    logits_difference = np.abs(primary_values - reference_logits)
    hidden_difference = np.abs(
        _bits_to_f16(primary_terminal).astype(np.float64)
        - reference_terminal.detach().cpu().numpy()
    )
    hidden_max = float(hidden_difference.max())
    logits_max = float(logits_difference.max())
    primary_argmax = int(np.argmax(primary_values))
    reference_argmax = int(np.argmax(reference_logits))
    if enforce_tolerances:
        _dialogue_require(
            hidden_max <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
            (
                f"step {ordinal} terminal hidden tolerance exceeded: "
                f"{hidden_max} > {TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE}"
            ),
        )
        _dialogue_require(
            logits_max <= LOGITS_ABSOLUTE_TOLERANCE,
            (
                f"step {ordinal} logits tolerance exceeded: "
                f"{logits_max} > {LOGITS_ABSOLUTE_TOLERANCE}"
            ),
        )
    _dialogue_require(
        primary_argmax == reference_argmax,
        (
            f"step {ordinal} argmax mismatch: primary={primary_argmax} "
            f"reference={reference_argmax}"
        ),
    )
    host_decision = Model24TokenDecisionHost(tokenizer).decide(primary_logits)
    _dialogue_require(
        host_decision["argmax_token_id"] == primary_argmax,
        f"step {ordinal} token host argmax mismatch",
    )
    content_ids = [
        token_id
        for token_id in (*generated_ids, primary_argmax)
        if token_id != EOS_TOKEN_ID
    ]
    return {
        "ordinal": ordinal,
        "decision_position": cache["position_count"] - 1,
        "cache_lineage": cache,
        "terminal_hidden": {
            "sha256": _sha256_bytes(_canonical_bytes(primary_terminal)),
            "independent_reference": {
                "implementation": "PyTorch CPU float64 dequantized-AWQ Qwen2",
                "max_abs_error": hidden_max,
                "mean_abs_error": float(hidden_difference.mean()),
                "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
                "within_tolerance": (
                    hidden_max <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE
                ),
            },
        },
        "logits": {
            "dtype": "F16",
            "vocab_size": len(primary_logits),
            "sha256": _sha256_bytes(
                _canonical_bytes(np.asarray(primary_logits, dtype="<u2"))
            ),
            "independent_reference_sha256": _sha256_bytes(
                _canonical_bytes(np.asarray(reference_logits, dtype="<f8"))
            ),
            "independent_reference": {
                "implementation": "PyTorch CPU float64 dequantized-AWQ tied head",
                "max_abs_error": logits_max,
                "mean_abs_error": float(logits_difference.mean()),
                "absolute_tolerance": LOGITS_ABSOLUTE_TOLERANCE,
                "within_tolerance": logits_max <= LOGITS_ABSOLUTE_TOLERANCE,
            },
        },
        "token": {
            **host_decision,
            "independent_reference_argmax_token_id": reference_argmax,
            "argmax_matches_independent_reference": True,
            "decoded_text_after_step": tokenizer.decode(
                content_ids,
                skip_special_tokens=False,
            ),
        },
    }


def execute_loaded_prompt(
    checkpoint_path: Path,
    tokenizer: Any,
    embeddings: np.ndarray,
    final_norm: np.ndarray,
    reference_lm_head: torch.Tensor,
    states: Sequence[LayerState],
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    enforce_tolerances: bool = True,
) -> dict[str, Any]:
    _dialogue_require(
        type(max_new_tokens) is int and max_new_tokens > 0,
        "max_new_tokens must be a positive integer",
    )
    _dialogue_require(
        bool(prompt_ids)
        and all(type(token_id) is int for token_id in prompt_ids),
        "prompt token IDs must be a non-empty integer sequence",
    )
    _reset_kv_caches(states)
    prompt_indices = np.asarray(prompt_ids, dtype=np.int64)
    primary_hidden = _f16_to_bits(embeddings[prompt_indices])
    reference_hidden = reference_lm_head[prompt_indices]
    primary_hidden, reference_hidden = _execute_positions(
        states,
        primary_hidden,
        reference_hidden,
        0,
    )

    generated_ids: list[int] = []
    steps = []
    previous_cache = None
    stop_reason = None
    with safe_open(checkpoint_path, framework="np") as checkpoint:
        while stop_reason is None:
            cache = _cache_snapshot(states, previous_cache)
            record = _decision_record(
                checkpoint,
                checkpoint_path,
                tokenizer,
                final_norm,
                reference_lm_head,
                primary_hidden[-1],
                reference_hidden[-1],
                cache,
                len(steps),
                generated_ids,
                enforce_tolerances=enforce_tolerances,
            )
            token_id = record["token"]["argmax_token_id"]
            generated_ids.append(token_id)
            steps.append(record)
            stop_reason = generation_stop_reason(
                token_id,
                len(generated_ids),
                max_new_tokens,
            )
            if stop_reason is not None:
                break
            position = len(prompt_ids) + len(generated_ids) - 1
            token_indices = np.asarray([token_id], dtype=np.int64)
            primary_hidden = _f16_to_bits(embeddings[token_indices])
            reference_hidden = reference_lm_head[token_indices]
            primary_hidden, reference_hidden = _execute_positions(
                states,
                primary_hidden,
                reference_hidden,
                position,
            )
            previous_cache = cache

    decoded_ids = [
        token_id for token_id in generated_ids if token_id != EOS_TOKEN_ID
    ]
    decoded_text = tokenizer.decode(decoded_ids, skip_special_tokens=False)
    return {
        "max_new_tokens": max_new_tokens,
        "stop_reason": stop_reason,
        "eos_token_id": EOS_TOKEN_ID,
        "eos_emitted": generated_ids[-1] == EOS_TOKEN_ID,
        "generated_token_ids": generated_ids,
        "decoded_token_ids": decoded_ids,
        "decoded_text": decoded_text,
        "decoded_utf8_sha256": _sha256_bytes(decoded_text.encode("utf-8")),
        "steps": steps,
    }


def execute_dialogue(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    _dialogue_require(
        type(max_new_tokens) is int and max_new_tokens > 0,
        "max_new_tokens must be a positive integer",
    )
    try:
        authenticate_checkpoint(checkpoint_path)
    except Exception as error:
        raise DialogueExecutionError(
            f"official checkpoint authentication failed: {error}"
        ) from error
    binding = _authenticate_model24_binding(_binding_path())
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    messages = [
        {"role": role, "content": content}
        for role, content in FIXED_CHAT_MESSAGES
    ]
    prompt = serialize_chat_prompt(messages)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    _dialogue_require(
        prompt == FIXED_CHAT_SERIALIZATION
        and prompt_ids == list(FIXED_CHAT_TOKEN_IDS),
        "fixed official dialogue prompt mismatch",
    )

    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    embeddings, final_norm, reference_lm_head, states = _load_model(
        checkpoint_path
    )
    generation = execute_loaded_prompt(
        checkpoint_path,
        tokenizer,
        embeddings,
        final_norm,
        reference_lm_head,
        states,
        prompt_ids,
        max_new_tokens=max_new_tokens,
    )
    return {
        "schema_version": 1,
        "kind": EVIDENCE_KIND,
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint": {
                "filename": checkpoint_path.name,
                "sha256": CHECKPOINT_SHA256,
                "bytes": CHECKPOINT_SIZE,
            },
            "accepted_model24_execution_binding": binding,
            "authenticated_layer_count": LAYER_COUNT,
            "authenticated_layer_tensor_count": (
                LAYER_COUNT * LAYER_TENSOR_COUNT
            ),
            "tied_lm_head": "model.embed_tokens.weight",
        },
        "tokenizer_binding": {
            "tokenizer_sha256": TOKENIZER_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "eos_token_id": EOS_TOKEN_ID,
        },
        "prompt": {
            "messages": messages,
            "serialization": prompt,
            "serialization_utf8_sha256": _sha256_bytes(prompt.encode("utf-8")),
            "token_ids": prompt_ids,
            "decoded_roundtrip": tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
            ),
        },
        "numeric_profile": {
            "primary": (
                "native asymmetric packed INT4 AWQ W4A16 G128 with exact "
                "ACE-3 FP16 operators and FP16 KV"
            ),
            "independent_reference": (
                "PyTorch CPU float64 dequantized-AWQ Qwen2 with causal KV"
            ),
            "projection_qzero_adjustment": "none",
            "selection": "greedy full-vocabulary argmax, lowest token ID tie-break",
        },
        "generation": generation,
        "claim_boundary": {
            "demonstrated": (
                "one deterministic greedy multi-token dialogue for the fixed "
                "authenticated prompt and checkpoint in the software/oracle executor"
            ),
            "broader_quality": "not assessed beyond the fixed prompt",
            "rtl": "full 24-layer dialogue execution not demonstrated in RTL",
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
            "latency": "not measured",
            "throughput": "not measured",
        },
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_document(
    document: Mapping[str, Any],
    tokenizer: Any | None = None,
    *,
    expected_kind: str = EVIDENCE_KIND,
    expected_prompt_serialization: str | None = FIXED_CHAT_SERIALIZATION,
    expected_prompt_token_ids: Sequence[int] | None = FIXED_CHAT_TOKEN_IDS,
    require_tolerances: bool = True,
) -> dict[str, Any]:
    _dialogue_require(document.get("schema_version") == 1, "schema version mismatch")
    _dialogue_require(document.get("kind") == expected_kind, "evidence kind mismatch")
    model = document["model_binding"]
    checkpoint = model["checkpoint"]
    _dialogue_require(
        checkpoint["sha256"] == CHECKPOINT_SHA256
        and checkpoint["bytes"] == CHECKPOINT_SIZE,
        "checkpoint binding mismatch",
    )
    binding = model["accepted_model24_execution_binding"]
    _dialogue_require(
        binding["sha256"] == MODEL24_BINDING_SHA256
        and binding["checkpoint_sha256"] == CHECKPOINT_SHA256,
        "accepted Model24 execution binding mismatch",
    )
    prompt = document["prompt"]
    _dialogue_require(
        bool(prompt["token_ids"])
        and all(type(token_id) is int for token_id in prompt["token_ids"])
        and prompt["serialization_utf8_sha256"]
        == _sha256_bytes(prompt["serialization"].encode("utf-8")),
        "prompt evidence mismatch",
    )
    if expected_prompt_serialization is not None:
        _dialogue_require(
            prompt["serialization"] == expected_prompt_serialization,
            "fixed prompt serialization mismatch",
        )
    if expected_prompt_token_ids is not None:
        _dialogue_require(
            prompt["token_ids"] == list(expected_prompt_token_ids),
            "fixed prompt token IDs mismatch",
        )
    if tokenizer is not None:
        _dialogue_require(
            tokenizer.encode(
                prompt["serialization"],
                add_special_tokens=False,
            ).ids
            == prompt["token_ids"],
            "official tokenizer prompt encoding mismatch",
        )
        _dialogue_require(
            tokenizer.decode(
                prompt["token_ids"],
                skip_special_tokens=False,
            )
            == prompt["decoded_roundtrip"],
            "official tokenizer prompt decode mismatch",
        )
    generation = document["generation"]
    steps = generation["steps"]
    max_new_tokens = generation["max_new_tokens"]
    _dialogue_require(
        type(max_new_tokens) is int
        and max_new_tokens > 0
        and 0 < len(steps) <= max_new_tokens,
        "generation step count is outside bounds",
    )
    generated_ids = []
    previous_cache = None
    for ordinal, step in enumerate(steps):
        _dialogue_require(step["ordinal"] == ordinal, "step ordering mismatch")
        cache = step["cache_lineage"]
        expected_count = len(prompt["token_ids"]) + ordinal
        _dialogue_require(
            cache["position_count"] == expected_count
            and step["decision_position"] == expected_count - 1,
            f"step {ordinal} cache position mismatch",
        )
        expected_parent_count = 0 if ordinal == 0 else expected_count - 1
        _dialogue_require(
            cache["parent_position_count"] == expected_parent_count,
            f"step {ordinal} cache parent count mismatch",
        )
        _dialogue_require(
            cache["added_positions"]
            == list(range(expected_parent_count, expected_count)),
            f"step {ordinal} cache did not append required positions",
        )
        _dialogue_require(
            len(cache["layers"]) == LAYER_COUNT,
            f"step {ordinal} cache does not cover 24 layers",
        )
        if previous_cache is None:
            _dialogue_require(
                cache["parent_cache_sha256"] is None,
                "first cache has a parent",
            )
        else:
            _dialogue_require(
                cache["parent_cache_sha256"] == previous_cache["cache_sha256"],
                f"step {ordinal} aggregate cache parentage mismatch",
            )
        prior_layers = (
            {}
            if previous_cache is None
            else {
                layer["layer_id"]: layer
                for layer in previous_cache["layers"]
            }
        )
        aggregate_layers = []
        for layer_id, layer in enumerate(cache["layers"]):
            _dialogue_require(
                layer["layer_id"] == layer_id,
                f"step {ordinal} cache layer ordering mismatch",
            )
            for name in (
                "k_sha256",
                "v_sha256",
                "appended_k_row_sha256",
                "appended_v_row_sha256",
            ):
                _dialogue_require(
                    _is_sha256(layer[name]),
                    f"step {ordinal} layer {layer_id} invalid {name}",
                )
            if previous_cache is None:
                _dialogue_require(
                    layer["parent_k_sha256"] is None
                    and layer["parent_v_sha256"] is None,
                    "first cache layer has parents",
                )
            else:
                prior = prior_layers[layer_id]
                _dialogue_require(
                    layer["parent_k_sha256"] == prior["k_sha256"]
                    and layer["parent_v_sha256"] == prior["v_sha256"],
                    f"step {ordinal} layer {layer_id} KV parentage mismatch",
                )
            aggregate_layers.append(
                {
                    "layer_id": layer_id,
                    "k_sha256": layer["k_sha256"],
                    "v_sha256": layer["v_sha256"],
                }
            )
        expected_cache_hash = _sha256_bytes(
            _canonical_json(
                {
                    "position_count": expected_count,
                    "layers": aggregate_layers,
                }
            )
        )
        _dialogue_require(
            cache["cache_sha256"] == expected_cache_hash,
            f"step {ordinal} aggregate cache hash mismatch",
        )
        hidden_comparison = step["terminal_hidden"]["independent_reference"]
        _dialogue_require(
            hidden_comparison["absolute_tolerance"]
            == TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE
            and hidden_comparison["within_tolerance"]
            == (
                hidden_comparison["max_abs_error"]
                <= TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE
            )
            and (
                not require_tolerances
                or hidden_comparison["within_tolerance"]
            ),
            f"step {ordinal} terminal hidden comparison failed",
        )
        comparison = step["logits"]["independent_reference"]
        _dialogue_require(
            comparison["absolute_tolerance"] == LOGITS_ABSOLUTE_TOLERANCE
            and comparison["within_tolerance"]
            == (comparison["max_abs_error"] <= LOGITS_ABSOLUTE_TOLERANCE)
            and (not require_tolerances or comparison["within_tolerance"]),
            f"step {ordinal} logits comparison failed",
        )
        token = step["token"]
        _dialogue_require(
            token["argmax_matches_independent_reference"]
            and token["argmax_token_id"]
            == token["independent_reference_argmax_token_id"],
            f"step {ordinal} argmax comparison failed",
        )
        generated_ids.append(token["argmax_token_id"])
        previous_cache = cache

    expected_reason = generation_stop_reason(
        generated_ids[-1],
        len(generated_ids),
        max_new_tokens,
    )
    _dialogue_require(
        expected_reason is not None
        and generation["stop_reason"] == expected_reason,
        "official stop handling mismatch",
    )
    _dialogue_require(
        generation["generated_token_ids"] == generated_ids,
        "generated token stream mismatch",
    )
    _dialogue_require(
        all(token_id != EOS_TOKEN_ID for token_id in generated_ids[:-1]),
        "generation continued after EOS",
    )
    decoded_ids = [
        token_id for token_id in generated_ids if token_id != EOS_TOKEN_ID
    ]
    _dialogue_require(
        generation["decoded_token_ids"] == decoded_ids
        and generation["decoded_utf8_sha256"]
        == _sha256_bytes(generation["decoded_text"].encode("utf-8"))
        and steps[-1]["token"]["decoded_text_after_step"]
        == generation["decoded_text"],
        "decoded output consistency mismatch",
    )
    _dialogue_require(
        generation["eos_token_id"] == EOS_TOKEN_ID
        and generation["eos_emitted"] == (generated_ids[-1] == EOS_TOKEN_ID),
        "EOS evidence mismatch",
    )
    if tokenizer is not None:
        _dialogue_require(
            tokenizer.decode(decoded_ids, skip_special_tokens=False)
            == generation["decoded_text"],
            "official tokenizer dialogue decode mismatch",
        )
        for step in steps:
            prefix = [
                token_id
                for token_id in generated_ids[: step["ordinal"] + 1]
                if token_id != EOS_TOKEN_ID
            ]
            _dialogue_require(
                step["token"]["decoded_text_after_step"]
                == tokenizer.decode(prefix, skip_special_tokens=False),
                f"step {step['ordinal']} decoded prefix mismatch",
            )
    claims = document["claim_boundary"]
    for name in (
        "broader_quality",
        "rtl",
        "synthesis",
        "ppa",
        "fpga",
        "latency",
        "throughput",
    ):
        _dialogue_require(name in claims, f"missing {name} claim boundary")
    return {
        "steps": len(steps),
        "generated_token_ids": generated_ids,
        "decoded_text": generation["decoded_text"],
        "stop_reason": generation["stop_reason"],
        "cache_positions": steps[-1]["cache_lineage"]["position_count"],
    }


def generate(
    output_dir: Path,
    checkpoint_path: Path = DEFAULT_OFFICIAL_CHECKPOINT,
    tokenizer_dir: Path = DEFAULT_OFFICIAL_TOKENIZER_DIR,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, bytes]:
    document = execute_dialogue(
        checkpoint_path,
        tokenizer_dir,
        max_new_tokens=max_new_tokens,
    )
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    summary = validate_document(document, tokenizer)
    evidence_payload = _canonical_json(document)
    manifest = {
        "schema_version": 1,
        "kind": "ace3_official_model24_multitoken_dialogue_manifest",
        "artifacts": {
            ARTIFACT_NAME: {
                "bytes": len(evidence_payload),
                "sha256": _sha256_bytes(evidence_payload),
            }
        },
        "summary": summary,
    }
    payloads = {
        ARTIFACT_NAME: evidence_payload,
        MANIFEST_NAME: _canonical_json(manifest),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.iterdir()}
    _dialogue_require(
        existing <= set(payloads),
        f"output directory contains unexpected files: {sorted(existing - set(payloads))}",
    )
    for name, payload in payloads.items():
        (output_dir / name).write_bytes(payload)
    return payloads


def validate_directory(
    vector_dir: Path,
    checkpoint_path: Path = DEFAULT_OFFICIAL_CHECKPOINT,
    tokenizer_dir: Path = DEFAULT_OFFICIAL_TOKENIZER_DIR,
) -> dict[str, Any]:
    _dialogue_require(vector_dir.is_dir(), f"evidence directory is missing: {vector_dir}")
    actual_names = {path.name for path in vector_dir.iterdir() if path.is_file()}
    _dialogue_require(
        actual_names == {ARTIFACT_NAME, MANIFEST_NAME},
        "evidence artifact set mismatch",
    )
    evidence_payload = (vector_dir / ARTIFACT_NAME).read_bytes()
    manifest = _json_without_duplicates(
        (vector_dir / MANIFEST_NAME).read_bytes(),
        MANIFEST_NAME,
    )
    record = manifest["artifacts"][ARTIFACT_NAME]
    _dialogue_require(
        record["bytes"] == len(evidence_payload)
        and record["sha256"] == _sha256_bytes(evidence_payload),
        "evidence manifest authentication failed",
    )
    try:
        authenticate_checkpoint(checkpoint_path)
    except Exception as error:
        raise DialogueExecutionError(
            f"official checkpoint authentication failed: {error}"
        ) from error
    _authenticate_model24_binding(_binding_path())
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    document = _json_without_duplicates(evidence_payload, ARTIFACT_NAME)
    summary = validate_document(document, tokenizer)
    _dialogue_require(summary == manifest["summary"], "manifest summary mismatch")
    with TemporaryDirectory(prefix="ace3-dialogue-validation-") as temporary:
        regenerated = generate(
            Path(temporary),
            checkpoint_path,
            tokenizer_dir,
            max_new_tokens=document["generation"]["max_new_tokens"],
        )
    _dialogue_require(
        all(
            regenerated[name] == (vector_dir / name).read_bytes()
            for name in regenerated
        ),
        "independent evidence regeneration mismatch",
    )
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
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.operation == "generate":
            _dialogue_require(args.output_dir is not None, "--output-dir is required")
            payloads = generate(
                args.output_dir.resolve(),
                args.official_checkpoint.resolve(),
                args.official_tokenizer_dir.resolve(),
                max_new_tokens=args.max_new_tokens,
            )
            document = json.loads(payloads[ARTIFACT_NAME])
            summary = validate_document(document)
            print(
                "OFFICIAL_MODEL24_DIALOGUE_GENERATION_PASS "
                f"steps={summary['steps']} "
                f"token_ids={summary['generated_token_ids']} "
                f"decoded_text={summary['decoded_text']!r} "
                f"stop={summary['stop_reason']}"
            )
        else:
            _dialogue_require(args.vector_dir is not None, "--vector-dir is required")
            summary = validate_directory(
                args.vector_dir.resolve(),
                args.official_checkpoint.resolve(),
                args.official_tokenizer_dir.resolve(),
            )
            print(
                "OFFICIAL_MODEL24_DIALOGUE_VALIDATION_PASS "
                f"steps={summary['steps']} "
                f"token_ids={summary['generated_token_ids']} "
                f"decoded_text={summary['decoded_text']!r} "
                f"stop={summary['stop_reason']}"
            )
    except (DialogueExecutionError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"OFFICIAL_MODEL24_DIALOGUE_FAIL {error}") from error


if __name__ == "__main__":
    main()
