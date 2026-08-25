#!/usr/bin/env python3
"""Deterministic official-checkpoint execution of one Qwen2.5 decoder layer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from safetensors import safe_open

from attention_oracle import attention_score, attention_softmax, attention_value
from awq_bit_oracle import AWQ_REVERSE_ORDER, q47_48_to_f16
from fp16_adaptation_oracle import (
    decode_f16_q24,
    residual_add,
    rmsnorm,
    silu_gate,
)
from model24_oracle import (
    CHECKPOINT_SHA256,
    ContractError as CheckpointContractError,
    authenticate_checkpoint,
)
from projection_oracle import complete_projection_output
from qwen2_rope_oracle import qwen2_coefficient, rotate_pair

MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
TOKEN_IDS = (9707, 1879)
TOKEN_TEXT = ("Hello", " world")
HIDDEN_SIZE = 896
INTERMEDIATE_SIZE = 4864
QUERY_HEADS = 14
KEY_VALUE_HEADS = 2
HEAD_DIM = 64
GROUP_SIZE = 128
ROPE_THETA = 1_000_000.0
EMBED_TENSOR_SHA256 = (
    "d74257dc547b48be5ae7b93f1c9af072c0c42dbbb85503078e25c59cd09e68d0"
)
TOKEN_ROW_SHA256 = {
    9707: "150deee94cfa96d5e9342ca4e5041b2b662ab649e7174e4280a0a1b062d10d06",
    1879: "e53571236e61d5517340953dbd4ac3cb0cc2acb144ff42c8017d02f33bb308a8",
}
TENSOR_BINDINGS = {
    "model.layers.0.input_layernorm.weight": (
        "870fae45e6c73031d339e11391519f9e556484f64341ebf479d4602d15eddca0",
        (896,),
    ),
    "model.layers.0.mlp.down_proj.qweight": (
        "2c13a8d3f06d8f0fa631e440b966b51bed53428c13babd7cd5cc041d05445c0c",
        (4864, 112),
    ),
    "model.layers.0.mlp.down_proj.qzeros": (
        "2959715fdd0c1053f350f137a9e25e4edc01d503976ffb377f79cb12fb85d5ca",
        (38, 112),
    ),
    "model.layers.0.mlp.down_proj.scales": (
        "5462785bab49a23dc904c9bbaa377ee065fc4409e02134d1c749ac8080d4f0fd",
        (38, 896),
    ),
    "model.layers.0.mlp.gate_proj.qweight": (
        "6dfde12b9161488b4afc071c20b40f99bce51ebf475561c726469fc56b9cfd39",
        (896, 608),
    ),
    "model.layers.0.mlp.gate_proj.qzeros": (
        "b47a8551b4b3e383660db480fca4837043217bc16125cd0105f2e207e342b375",
        (7, 608),
    ),
    "model.layers.0.mlp.gate_proj.scales": (
        "c0293a55853e091765697c3aeeecedca1d6762826e347de0baf0bb09ead66d5c",
        (7, 4864),
    ),
    "model.layers.0.mlp.up_proj.qweight": (
        "e79dc28b77e8585fba8892138043e6267d409ddb553cd08eaebb481f4757b947",
        (896, 608),
    ),
    "model.layers.0.mlp.up_proj.qzeros": (
        "f183566f407956ea547b14cf7d25cd274ec162b220f833f359d20f7a88c27086",
        (7, 608),
    ),
    "model.layers.0.mlp.up_proj.scales": (
        "4e670cb7c50c8aa0ebde0ddbd2d11917b8e1e40410d212f385da8e94f9562361",
        (7, 4864),
    ),
    "model.layers.0.post_attention_layernorm.weight": (
        "fe1626b78ab16da772646b9447b325e8f7186689812203fe766c39cd342f48ac",
        (896,),
    ),
    "model.layers.0.self_attn.k_proj.bias": (
        "f2e734d74baaf83323766390c88032ba71e2c3450b49ad59bff48cb1b6153b14",
        (128,),
    ),
    "model.layers.0.self_attn.k_proj.qweight": (
        "317c4bf89c6c520ae370d8866c0758d5c13b9478457e75b88560874a3e0b4919",
        (896, 16),
    ),
    "model.layers.0.self_attn.k_proj.qzeros": (
        "a147e8a7ee49fc1b9c66616edf91eaa3fd835bd3f0ea8663385e9c44ac81542d",
        (7, 16),
    ),
    "model.layers.0.self_attn.k_proj.scales": (
        "c1ba623cd0e6f3427610636813f6f86c375d8bf142a3080b5e32c6b13b5a1df8",
        (7, 128),
    ),
    "model.layers.0.self_attn.o_proj.qweight": (
        "458319c35aafc2c5db11b34c1b8c364ee99bb0288460632804cd20c8df38d1ab",
        (896, 112),
    ),
    "model.layers.0.self_attn.o_proj.qzeros": (
        "fa284a49a5a5801248866428762d74db731259c79d386e57cdf77f51779c8505",
        (7, 112),
    ),
    "model.layers.0.self_attn.o_proj.scales": (
        "324314c1c80e8247d5fc4476479bcdc8d875124aab356a7c33eb24aa157ff5fc",
        (7, 896),
    ),
    "model.layers.0.self_attn.q_proj.bias": (
        "e9612c72520a62dd903796d9535de2081e1cd15a724b2972513599ff276b9e72",
        (896,),
    ),
    "model.layers.0.self_attn.q_proj.qweight": (
        "db4770023698611ff0115d220590fdb8232fbe5dcbd22fbe80e0bcdc838caf87",
        (896, 112),
    ),
    "model.layers.0.self_attn.q_proj.qzeros": (
        "3cf7cd5712dd7523db3c7dd47c2b1d582e19545036f75b95ff0331c1fc0c596c",
        (7, 112),
    ),
    "model.layers.0.self_attn.q_proj.scales": (
        "687adc7d7bcd6e45a065f914dd27a1284b7e48260491bb0d26ae1e13b78ac321",
        (7, 896),
    ),
    "model.layers.0.self_attn.v_proj.bias": (
        "ab149ae196a853c3ba8a632b5c2967f863791c4a154ebeeeb25924e71ebb6e77",
        (128,),
    ),
    "model.layers.0.self_attn.v_proj.qweight": (
        "10489a051595da56b98922ee012cf4758802a4b2cb791d9f697d7d6845681a80",
        (896, 16),
    ),
    "model.layers.0.self_attn.v_proj.qzeros": (
        "17f6395454631f98f7fb6acfbf5ffbaf56de9a04131b9c46d2b3ed22a39e8942",
        (7, 16),
    ),
    "model.layers.0.self_attn.v_proj.scales": (
        "ac75b583638863504667fc755b9f42b2ce85437ef02a79f428f11bac6b739d00",
        (7, 128),
    ),
}

REFERENCE_TOLERANCES = {
    "input_rmsnorm": 0.02,
    "q_proj": 0.1,
    "k_proj": 0.1,
    "v_proj": 0.1,
    "q_rope": 0.15,
    "k_rope": 0.15,
    "attention_scores": 0.2,
    "attention_probabilities": 0.08,
    "attention_value": 0.15,
    "o_proj": 0.2,
    "attention_residual": 0.2,
    "post_attention_rmsnorm": 0.2,
    "gate_proj": 0.3,
    "up_proj": 0.3,
    "silu": 0.55,
    "down_proj": 0.5,
    "post_layer_hidden": 0.5,
}


class LayerExecutionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LayerExecutionError(message)


def _canonical_bytes(array: np.ndarray) -> bytes:
    dtype = array.dtype.newbyteorder("<")
    return np.ascontiguousarray(array, dtype=dtype).tobytes()


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(_canonical_bytes(array)).hexdigest()


def _f16_to_bits(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype="<f2").view("<u2")


def _bits_to_f16(bits: np.ndarray | list[int]) -> np.ndarray:
    return np.ascontiguousarray(bits, dtype="<u2").view("<f2")


def _bits_to_q24(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint16)
    exponent = (values >> 10) & 0x1F
    fraction = values & 0x3FF
    _require(not np.any(exponent == 0x1F), "nonfinite FP16 activation")
    magnitude = np.where(
        exponent == 0,
        fraction.astype(np.int64),
        (0x400 | fraction).astype(np.int64)
        * np.left_shift(np.int64(1), np.maximum(exponent.astype(np.int64) - 1, 0)),
    )
    return np.where(values & 0x8000, -magnitude, magnitude)


def _unpack_words(words: np.ndarray) -> np.ndarray:
    unsigned = np.asarray(words, dtype=np.int32).view(np.uint32)
    lanes = [((unsigned >> (4 * physical)) & 0xF).astype(np.int64)
             for physical in AWQ_REVERSE_ORDER]
    return np.stack(lanes, axis=-1).reshape(*unsigned.shape[:-1], unsigned.shape[-1] * 8)


def _load_tensors(checkpoint_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    try:
        authenticate_checkpoint(checkpoint_path)
    except CheckpointContractError as error:
        raise LayerExecutionError(f"official checkpoint authentication failed: {error}") from error
    tensors: dict[str, np.ndarray] = {}
    with safe_open(checkpoint_path, framework="np") as checkpoint:
        for name, (expected_hash, expected_shape) in TENSOR_BINDINGS.items():
            value = np.asarray(checkpoint.get_tensor(name))
            _require(tuple(value.shape) == expected_shape, f"{name} shape mismatch")
            _require(_sha256(value) == expected_hash, f"{name} SHA256 mismatch")
            tensors[name] = value
        embedding_slice = checkpoint.get_slice("model.embed_tokens.weight")
        embeddings = np.stack(
            [np.asarray(embedding_slice[token_id], dtype="<f2") for token_id in TOKEN_IDS]
        )
    for token_id, row in zip(TOKEN_IDS, embeddings, strict=True):
        _require(_sha256(row) == TOKEN_ROW_SHA256[token_id],
                 f"embedding row {token_id} SHA256 mismatch")
    return tensors, embeddings


def _projection(
    activation_bits: np.ndarray,
    tensors: dict[str, np.ndarray],
    prefix: str,
    bias_name: str | None = None,
) -> tuple[np.ndarray, int]:
    qweight = tensors[f"{prefix}.qweight"]
    qzeros = tensors[f"{prefix}.qzeros"]
    scales = tensors[f"{prefix}.scales"]
    quantized = _unpack_words(qweight)
    zeros = _unpack_words(qzeros)
    scale_q24 = _bits_to_q24(_f16_to_bits(scales))
    activation_q24 = _bits_to_q24(activation_bits)
    groups = activation_bits.size // GROUP_SIZE
    out_features = quantized.shape[1]
    _require(
        quantized.shape == (activation_bits.size, out_features)
        and zeros.shape == (groups, out_features)
        and scales.shape == (groups, out_features),
        f"{prefix} AWQ geometry mismatch",
    )
    accumulator = [0] * out_features
    for group in range(groups):
        begin = group * GROUP_SIZE
        end = begin + GROUP_SIZE
        delta = quantized[begin:end] - zeros[group]
        group_dot = activation_q24[begin:end] @ delta
        for channel in range(out_features):
            accumulator[channel] += int(group_dot[channel]) * int(scale_q24[group, channel])
    output = np.asarray([q47_48_to_f16(value)[0] for value in accumulator], dtype="<u2")
    bias = None if bias_name is None else _f16_to_bits(tensors[bias_name])
    if bias is not None:
        output = np.asarray(
            [residual_add(int(value), int(bias[index]))[0]
             for index, value in enumerate(output)],
            dtype="<u2",
        )

    sampled = 0
    words = out_features // 8
    for channel in sorted({0, out_features // 2, out_features - 1}):
        lane = channel & 7
        packed = channel >> 3
        bit_result = complete_projection_output(
            activation_bits.tolist(),
            [int(qweight[index, packed]) for index in range(activation_bits.size)],
            [int(qzeros[group, packed]) for group in range(groups)],
            [int(_f16_to_bits(scales)[group, channel]) for group in range(groups)],
            lane,
            None if bias is None else int(bias[channel]),
        )
        _require(not bit_result[2], f"{prefix} sampled bit oracle invalid")
        _require(bit_result[1] == int(output[channel]),
                 f"{prefix} sampled bit oracle mismatch at channel {channel}")
        sampled += 1
    _require(words * 8 == out_features, f"{prefix} packed output geometry mismatch")
    return output, sampled


def _finite_rmsnorm(activation: np.ndarray, weight: np.ndarray) -> np.ndarray:
    outputs, _, _ = rmsnorm(activation.tolist(), _f16_to_bits(weight).tolist())
    _require(not any(invalid for _, invalid, _ in outputs), "RMSNorm produced invalid output")
    return np.asarray([value for value, _, _ in outputs], dtype="<u2")


def _rope(vector: np.ndarray, heads: int, position: int) -> np.ndarray:
    result = vector.copy()
    for head in range(heads):
        base = head * HEAD_DIM
        for pair in range(HEAD_DIM // 2):
            cosine, sine = qwen2_coefficient(position, pair)
            low, high, invalid, _ = rotate_pair(
                int(vector[base + pair]),
                int(vector[base + pair + HEAD_DIM // 2]),
                cosine,
                sine,
            )
            _require(not invalid, "RoPE produced invalid output")
            result[base + pair] = low
            result[base + pair + HEAD_DIM // 2] = high
    return result


def _residual(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = []
    for left_item, right_item in zip(left, right, strict=True):
        value, invalid, _ = residual_add(int(left_item), int(right_item))
        _require(not invalid, "residual produced invalid output")
        output.append(value)
    return np.asarray(output, dtype="<u2")


def _primary_execution(
    tensors: dict[str, np.ndarray],
    embeddings: np.ndarray,
) -> tuple[dict[str, np.ndarray], int]:
    stage_rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "embedding",
            "input_rmsnorm",
            "q_proj",
            "k_proj",
            "v_proj",
            "q_rope",
            "k_rope",
            "kv_cache_k",
            "kv_cache_v",
            "attention_scores",
            "attention_probabilities",
            "attention_value",
            "o_proj",
            "attention_residual",
            "post_attention_rmsnorm",
            "gate_proj",
            "up_proj",
            "silu",
            "down_proj",
            "post_layer_hidden",
        )
    }
    cache_k: list[np.ndarray] = []
    cache_v: list[np.ndarray] = []
    sampled_checks = 0
    for position, embedding in enumerate(embeddings):
        activation = _f16_to_bits(embedding)
        stage_rows["embedding"].append(activation)
        norm1 = _finite_rmsnorm(
            activation,
            tensors["model.layers.0.input_layernorm.weight"],
        )
        stage_rows["input_rmsnorm"].append(norm1)
        q, checks = _projection(
            norm1,
            tensors,
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.q_proj.bias",
        )
        sampled_checks += checks
        k, checks = _projection(
            norm1,
            tensors,
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.k_proj.bias",
        )
        sampled_checks += checks
        v, checks = _projection(
            norm1,
            tensors,
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.v_proj.bias",
        )
        sampled_checks += checks
        stage_rows["q_proj"].append(q)
        stage_rows["k_proj"].append(k)
        stage_rows["v_proj"].append(v)
        rotated_q = _rope(q, QUERY_HEADS, position)
        rotated_k = _rope(k, KEY_VALUE_HEADS, position)
        stage_rows["q_rope"].append(rotated_q.reshape(QUERY_HEADS, HEAD_DIM))
        stage_rows["k_rope"].append(rotated_k.reshape(KEY_VALUE_HEADS, HEAD_DIM))
        cache_k.append(rotated_k.reshape(KEY_VALUE_HEADS, HEAD_DIM))
        cache_v.append(v.reshape(KEY_VALUE_HEADS, HEAD_DIM))
        stage_rows["kv_cache_k"].append(cache_k[-1])
        stage_rows["kv_cache_v"].append(cache_v[-1])

        score_row = np.zeros((QUERY_HEADS, len(TOKEN_IDS)), dtype="<u2")
        probability_row = np.zeros_like(score_row)
        attended = np.zeros((QUERY_HEADS, HEAD_DIM), dtype="<u2")
        for query_head in range(QUERY_HEADS):
            kv_head = query_head // (QUERY_HEADS // KEY_VALUE_HEADS)
            scores = []
            for key_position in range(position + 1):
                score = attention_score(
                    rotated_q.reshape(QUERY_HEADS, HEAD_DIM)[query_head].tolist(),
                    cache_k[key_position][kv_head].tolist(),
                    [True] * HEAD_DIM,
                    position,
                    key_position,
                )
                _require(not score.invalid and not score.cache_miss,
                         "attention score produced invalid output")
                scores.append(score.score_f16)
                score_row[query_head, key_position] = score.score_f16
            probabilities = attention_softmax(
                scores,
                list(range(position + 1)),
                [True] * (position + 1),
                [False] * (position + 1),
                [False] * (position + 1),
                position,
            )
            _require(
                not probabilities.invalid
                and not probabilities.cache_miss
                and not probabilities.row_error,
                "attention softmax produced invalid output",
            )
            for key_position, probability in enumerate(probabilities.probabilities_f16):
                probability_row[query_head, key_position] = probability
            for dimension in range(HEAD_DIM):
                value = attention_value(
                    list(probabilities.probabilities_f16),
                    [int(cache_v[key_position][kv_head, dimension])
                     for key_position in range(position + 1)],
                    [True] * (position + 1),
                    [False] * (position + 1),
                )
                _require(
                    not value.invalid and not value.cache_miss and not value.row_error,
                    "attention value produced invalid output",
                )
                attended[query_head, dimension] = value.value_f16
        stage_rows["attention_scores"].append(score_row)
        stage_rows["attention_probabilities"].append(probability_row)
        stage_rows["attention_value"].append(attended)
        output, checks = _projection(
            attended.reshape(-1),
            tensors,
            "model.layers.0.self_attn.o_proj",
        )
        sampled_checks += checks
        stage_rows["o_proj"].append(output)
        residual1 = _residual(output, activation)
        stage_rows["attention_residual"].append(residual1)
        norm2 = _finite_rmsnorm(
            residual1,
            tensors["model.layers.0.post_attention_layernorm.weight"],
        )
        stage_rows["post_attention_rmsnorm"].append(norm2)
        gate, checks = _projection(norm2, tensors, "model.layers.0.mlp.gate_proj")
        sampled_checks += checks
        up, checks = _projection(norm2, tensors, "model.layers.0.mlp.up_proj")
        sampled_checks += checks
        stage_rows["gate_proj"].append(gate)
        stage_rows["up_proj"].append(up)
        activated = []
        for gate_item, up_item in zip(gate, up, strict=True):
            value, invalid, _ = silu_gate(int(gate_item), int(up_item))
            _require(not invalid, "SiLU gate produced invalid output")
            activated.append(value)
        silu = np.asarray(activated, dtype="<u2")
        stage_rows["silu"].append(silu)
        down, checks = _projection(silu, tensors, "model.layers.0.mlp.down_proj")
        sampled_checks += checks
        stage_rows["down_proj"].append(down)
        stage_rows["post_layer_hidden"].append(_residual(down, residual1))
    return {name: np.stack(rows) for name, rows in stage_rows.items()}, sampled_checks


def _torch_unpack(words: np.ndarray) -> torch.Tensor:
    unsigned = torch.from_numpy(np.asarray(words, dtype=np.int32).copy()).to(torch.int64)
    lanes = [torch.bitwise_and(torch.bitwise_right_shift(unsigned, 4 * physical), 0xF)
             for physical in AWQ_REVERSE_ORDER]
    return torch.stack(lanes, dim=-1).reshape(*unsigned.shape[:-1], unsigned.shape[-1] * 8)


def _torch_linear(
    activation: torch.Tensor,
    tensors: dict[str, np.ndarray],
    prefix: str,
    bias_name: str | None = None,
) -> torch.Tensor:
    quantized = _torch_unpack(tensors[f"{prefix}.qweight"]).to(torch.float64)
    zeros = _torch_unpack(tensors[f"{prefix}.qzeros"]).to(torch.float64)
    scales = torch.from_numpy(tensors[f"{prefix}.scales"].astype(np.float64))
    groups = activation.shape[-1] // GROUP_SIZE
    weight = (quantized - zeros.repeat_interleave(GROUP_SIZE, dim=0)) * (
        scales.repeat_interleave(GROUP_SIZE, dim=0)
    )
    output = activation.to(torch.float64) @ weight
    if bias_name is not None:
        output = output + torch.from_numpy(tensors[bias_name].astype(np.float64))
    _require(groups == scales.shape[0], f"{prefix} PyTorch group geometry mismatch")
    return output


def _torch_rmsnorm(
    activation: torch.Tensor,
    weight: np.ndarray,
) -> torch.Tensor:
    variance = activation.to(torch.float64).pow(2).mean(-1, keepdim=True)
    normalized = activation.to(torch.float64) * torch.rsqrt(variance + 1e-6)
    return normalized * torch.from_numpy(weight.astype(np.float64))


def _torch_reference(
    tensors: dict[str, np.ndarray],
    embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    torch.set_num_threads(1)
    activation = torch.from_numpy(embeddings.astype(np.float64))
    stages: dict[str, torch.Tensor] = {}
    norm1 = _torch_rmsnorm(
        activation,
        tensors["model.layers.0.input_layernorm.weight"],
    )
    stages["input_rmsnorm"] = norm1
    q = _torch_linear(
        norm1,
        tensors,
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.q_proj.bias",
    )
    k = _torch_linear(
        norm1,
        tensors,
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.k_proj.bias",
    )
    v = _torch_linear(
        norm1,
        tensors,
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.v_proj.bias",
    )
    stages["q_proj"], stages["k_proj"], stages["v_proj"] = q, k, v
    positions = torch.arange(len(TOKEN_IDS), dtype=torch.float64)
    frequencies = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
    )
    angles = torch.outer(positions, frequencies)
    cosine, sine = torch.cos(angles), torch.sin(angles)

    def rotate(values: torch.Tensor, heads: int) -> torch.Tensor:
        shaped = values.reshape(len(TOKEN_IDS), heads, HEAD_DIM)
        low, high = shaped[..., : HEAD_DIM // 2], shaped[..., HEAD_DIM // 2 :]
        rotated_low = low * cosine[:, None, :] - high * sine[:, None, :]
        rotated_high = high * cosine[:, None, :] + low * sine[:, None, :]
        return torch.cat((rotated_low, rotated_high), dim=-1)

    rotated_q = rotate(q, QUERY_HEADS)
    rotated_k = rotate(k, KEY_VALUE_HEADS)
    values = v.reshape(len(TOKEN_IDS), KEY_VALUE_HEADS, HEAD_DIM)
    stages["q_rope"], stages["k_rope"] = rotated_q, rotated_k
    scores = torch.zeros((len(TOKEN_IDS), QUERY_HEADS, len(TOKEN_IDS)), dtype=torch.float64)
    probabilities = torch.zeros_like(scores)
    attended = torch.zeros(
        (len(TOKEN_IDS), QUERY_HEADS, HEAD_DIM),
        dtype=torch.float64,
    )
    for position in range(len(TOKEN_IDS)):
        for query_head in range(QUERY_HEADS):
            kv_head = query_head // (QUERY_HEADS // KEY_VALUE_HEADS)
            row = (
                rotated_q[position, query_head]
                @ rotated_k[: position + 1, kv_head].T
                / (HEAD_DIM**0.5)
            )
            probability = torch.softmax(row, dim=-1)
            scores[position, query_head, : position + 1] = row
            probabilities[position, query_head, : position + 1] = probability
            attended[position, query_head] = (
                probability @ values[: position + 1, kv_head]
            )
    stages["attention_scores"] = scores
    stages["attention_probabilities"] = probabilities
    stages["attention_value"] = attended
    output = _torch_linear(
        attended.reshape(len(TOKEN_IDS), HIDDEN_SIZE),
        tensors,
        "model.layers.0.self_attn.o_proj",
    )
    stages["o_proj"] = output
    residual1 = output + activation
    stages["attention_residual"] = residual1
    norm2 = _torch_rmsnorm(
        residual1,
        tensors["model.layers.0.post_attention_layernorm.weight"],
    )
    stages["post_attention_rmsnorm"] = norm2
    gate = _torch_linear(norm2, tensors, "model.layers.0.mlp.gate_proj")
    up = _torch_linear(norm2, tensors, "model.layers.0.mlp.up_proj")
    stages["gate_proj"], stages["up_proj"] = gate, up
    activated = torch_functional.silu(gate) * up
    stages["silu"] = activated
    down = _torch_linear(activated, tensors, "model.layers.0.mlp.down_proj")
    stages["down_proj"] = down
    stages["post_layer_hidden"] = residual1 + down
    return {name: value.detach().cpu().numpy() for name, value in stages.items()}


def _stage_record(name: str, bits: np.ndarray) -> dict[str, Any]:
    payload = _canonical_bytes(bits.astype("<u2"))
    values = _bits_to_f16(bits).astype(np.float64)
    return {
        "name": name,
        "shape": list(bits.shape),
        "dtype": "F16",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "nonzero_count": int(np.count_nonzero(bits & 0x7FFF)),
        "finite": bool(np.all(np.isfinite(values))),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def official_single_decoder_layer_contract() -> dict[str, Any]:
    return {
        "artifact": "official_layer0_slice.json",
        "prompt": {
            "utf8": "Hello world",
            "token_ids": list(TOKEN_IDS),
            "positions": [0, 1],
        },
        "geometry": {
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "query_heads": QUERY_HEADS,
            "key_value_heads": KEY_VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "group_size": GROUP_SIZE,
        },
        "embedding_tensor_sha256": EMBED_TENSOR_SHA256,
        "embedding_row_sha256": {
            str(token_id): TOKEN_ROW_SHA256[token_id] for token_id in TOKEN_IDS
        },
        "consumed_layer_tensor_count": len(TENSOR_BINDINGS),
        "consumed_tensor_binding_location": (
            "official_layer0_slice.json.model_binding.consumed_tensors"
        ),
        "intermediate_stages": [
            "embedding",
            "input_rmsnorm",
            "q_proj",
            "k_proj",
            "v_proj",
            "q_rope",
            "k_rope",
            "kv_cache_k",
            "kv_cache_v",
            "attention_scores",
            "attention_probabilities",
            "attention_value",
            "o_proj",
            "attention_residual",
            "post_attention_rmsnorm",
            "gate_proj",
            "up_proj",
            "silu",
            "down_proj",
            "post_layer_hidden",
        ],
        "independent_reference": {
            "implementation": "PyTorch CPU float64",
            "absolute_tolerances": REFERENCE_TOLERANCES,
            "sampled_projection_bit_oracle_outputs": 42,
        },
        "handoff": {
            "shape": [HIDDEN_SIZE],
            "dtype": "F16",
            "compatible_interface": "model.norm.weight -> tied lm_head.weight",
            "requires_layers_before_final_head": list(range(1, 24)),
        },
    }


def official_single_decoder_layer_document(checkpoint_path: Path) -> dict[str, Any]:
    tensors, embeddings = _load_tensors(checkpoint_path)
    primary, sampled_checks = _primary_execution(tensors, embeddings)
    reference = _torch_reference(tensors, embeddings)
    comparisons: dict[str, dict[str, Any]] = {}
    for name, tolerance in REFERENCE_TOLERANCES.items():
        primary_values = _bits_to_f16(primary[name]).astype(np.float64)
        reference_values = reference[name].astype(np.float64)
        difference = np.abs(primary_values - reference_values)
        maximum = float(difference.max())
        comparisons[name] = {
            "reference": "independent PyTorch float64 dequantized-AWQ Qwen2 path",
            "max_abs_error": maximum,
            "mean_abs_error": float(difference.mean()),
            "absolute_tolerance": tolerance,
            "within_tolerance": maximum <= tolerance,
        }
        _require(maximum <= tolerance, f"{name} PyTorch tolerance exceeded: {maximum}")
    stage_records = [_stage_record(name, bits) for name, bits in primary.items()]
    second_probabilities = _bits_to_f16(primary["attention_probabilities"][1])
    final_bits = primary["post_layer_hidden"][1].astype("<u2")
    _require(
        all(record["finite"] and record["nonzero_count"] > 0 for record in stage_records),
        "layer execution produced a vacuous or nonfinite stage",
    )
    _require(
        np.all(second_probabilities[:, :2] > 0),
        "second token did not attend to both populated cache positions",
    )
    _require(
        not np.array_equal(primary["post_layer_hidden"], primary["embedding"]),
        "post-layer hidden state is vacuous",
    )
    tensor_records = [
        {
            "name": name,
            "shape": list(expected_shape),
            "dtype": (
                "F16"
                if tensors[name].dtype == np.float16
                else "I32"
            ),
            "bytes": len(_canonical_bytes(tensors[name])),
            "sha256": expected_hash,
        }
        for name, (expected_hash, expected_shape) in sorted(TENSOR_BINDINGS.items())
    ]
    return {
        "schema_version": 1,
        "kind": "ace3_official_single_decoder_layer",
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "consumed_tensors": tensor_records,
            "embedding_tensor_sha256": EMBED_TENSOR_SHA256,
            "embedding_rows": [
                {
                    "token_id": token_id,
                    "token": token,
                    "shape": [HIDDEN_SIZE],
                    "dtype": "F16",
                    "sha256": TOKEN_ROW_SHA256[token_id],
                }
                for token_id, token in zip(TOKEN_IDS, TOKEN_TEXT, strict=True)
            ],
        },
        "numeric_profile": {
            "projection": "native asymmetric packed INT4 AWQ W4A16 G128",
            "packed_nibble_order": list(AWQ_REVERSE_ORDER),
            "qzero_adjustment": "none",
            "projection_accumulator": "exact signed Q53.48, round once to F16",
            "activations": "F16",
            "scales": "F16",
            "kv_cache": "F16",
            "rope": "Qwen half-split, theta=1000000, F16 multiply/add",
            "attention": "14:2 GQA, causal two-position score/softmax/value",
            "silu": "accepted rational-RNE F16 adaptation primitive",
        },
        "prompt": {
            "utf8": "Hello world",
            "token_ids": list(TOKEN_IDS),
            "positions": [0, 1],
        },
        "intermediates": stage_records,
        "independent_reference": {
            "implementation": "PyTorch CPU float64",
            "semantics": (
                "dequantized official AWQ tensors with Qwen RMSNorm, RoPE, "
                "causal GQA softmax, and true SiLU"
            ),
            "comparisons": comparisons,
            "sampled_projection_bit_oracle_checks": sampled_checks,
        },
        "non_vacuity": {
            "all_stages_finite_and_nonzero": True,
            "second_token_reads_kv_positions": [0, 1],
            "second_token_all_heads_have_two_positive_probabilities": True,
            "post_layer_hidden_differs_from_embedding": True,
        },
        "final_token_decision_handoff": {
            "source": "post_layer_hidden[token_index=1]",
            "shape": [HIDDEN_SIZE],
            "dtype": "F16",
            "sha256": hashlib.sha256(_canonical_bytes(final_bits)).hexdigest(),
            "f16_bits": [int(value) for value in final_bits],
            "compatible_interface": "model.norm.weight -> tied lm_head.weight",
            "status": (
                "format-compatible only; layers 1 through 23 must execute before "
                "this state is a valid full-model terminal hidden state"
            ),
        },
        "claim_boundary": (
            "Authenticated deterministic numerical execution of layer 0 for two "
            "fixed tokens only. The PyTorch comparison uses documented tolerances "
            "because accepted RTL approximations differ from Qwen float operators. "
            "No layers 1 through 23, full-model token, dialogue, RTL simulation, "
            "synthesis, PPA, FPGA, latency, or throughput result is claimed."
        ),
    }
