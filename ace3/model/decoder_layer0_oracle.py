#!/usr/bin/env python3
"""Independent integer-oracle composition for one Qwen2.5 layer-0 token."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from attention_oracle import attention_score, attention_softmax, attention_value
from fp16_adaptation_oracle import residual_add, rmsnorm, silu_gate, silu_gate_exp
from projection_oracle import complete_projection_output
from qwen2_rope_oracle import qwen2_coefficient, rotate_pair

HIDDEN, INTERMEDIATE, HEADS, KV_HEADS, HEAD_DIM = 896, 4864, 14, 2, 64
MODEL = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"


def _read(path: Path, dtype: str) -> list[int]:
    raw = path.read_bytes()
    unit = 2 if dtype == "float16" else 4
    return list(struct.unpack(f"<{len(raw) // unit}{'H' if unit == 2 else 'I'}", raw))


def load_authenticated_tensors(root: Path) -> tuple[dict[str, list[int]], dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["model_repository"] != MODEL or manifest["model_revision"] != REVISION:
        raise RuntimeError("fixed model identity mismatch")
    values: dict[str, list[int]] = {}
    for item in manifest["tensors"] + manifest["token_embeddings"]:
        path = root / item["serialized_file"]
        raw = path.read_bytes()
        if len(raw) != item["byte_count"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise RuntimeError(f"tensor source hash mismatch: {path.name}")
        values[item["checkpoint_name"] + ":" + str(item.get("token_index", ""))] = _read(
            path, item["dtype"])
    return values, manifest


def _projection(activations: list[int], qweight: list[int], qzeros: list[int],
                scales: list[int], out_features: int, bias: list[int] | None) -> list[int]:
    groups, words = len(activations) // 128, out_features // 8
    if len(qweight) != len(activations) * words or len(qzeros) != groups * words:
        raise ValueError("AWQ tensor geometry")
    if len(scales) != groups * out_features:
        raise ValueError("AWQ scale geometry")
    output: list[int] = []
    for channel in range(out_features):
        lane, packed = channel & 7, channel >> 3
        qweight_column = [qweight[index * words + packed] for index in range(len(activations))]
        zero_column = [qzeros[group * words + packed] for group in range(groups)]
        scale_column = [scales[group * out_features + channel] for group in range(groups)]
        _, result, invalid, _, _ = complete_projection_output(
            activations, qweight_column, zero_column, scale_column, lane,
            None if bias is None else bias[channel])
        if invalid:
            raise ArithmeticError(f"nonfinite projection operand at channel {channel}")
        output.append(result)
    return output


def _module(values: dict[str, list[int]], prefix: str, activations: list[int],
            out_features: int, bias: list[int] | None = None) -> list[int]:
    return _projection(
        activations,
        values[f"model.layers.0.{prefix}.qweight:"],
        values[f"model.layers.0.{prefix}.qzeros:"],
        values[f"model.layers.0.{prefix}.scales:"],
        out_features, bias)


def _rope(vector: list[int], heads: int, position: int) -> list[int]:
    """Qwen's half-split rotary layout: [0:32] pairs with [32:64]."""
    rotated = vector.copy()
    for head in range(heads):
        base = head * HEAD_DIM
        for pair in range(HEAD_DIM // 2):
            cos, sin = qwen2_coefficient(position, pair)
            low, high, invalid, _ = rotate_pair(
                vector[base + pair], vector[base + pair + 32], cos, sin)
            if invalid:
                raise ArithmeticError("nonfinite RoPE operand")
            rotated[base + pair], rotated[base + pair + 32] = low, high
    return rotated


def _expect_finite(outputs: list[tuple[int, bool, bool]], name: str) -> list[int]:
    if any(invalid for _, invalid, _ in outputs):
        raise ArithmeticError(f"{name} invalid")
    return [value for value, _, _ in outputs]


def run_token(values: dict[str, list[int]], activation: list[int], position: int,
              cache_k: list[list[int]], cache_v: list[list[int]],
              accurate_silu: bool = False) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    """Run one token and return final vector plus (stage,index,f16,position) trace."""
    trace: list[tuple[int, int, int, int]] = []
    n1 = _expect_finite(rmsnorm(activation, values["model.layers.0.input_layernorm.weight:"])[0], "norm1")
    trace.extend((0, i, item, position) for i, item in enumerate(n1))
    q = _module(values, "self_attn.q_proj", n1, HIDDEN, values["model.layers.0.self_attn.q_proj.bias:"])
    trace.extend((1, i, item, position) for i, item in enumerate(q))
    rq = _rope(q, HEADS, position)
    for head in range(HEADS):
        for pair in range(32):
            base = head * HEAD_DIM
            trace.extend(((4, base + pair, rq[base + pair], position),
                          (4, base + pair + 32, rq[base + pair + 32], position)))
    k = _module(values, "self_attn.k_proj", n1, KV_HEADS * HEAD_DIM,
                values["model.layers.0.self_attn.k_proj.bias:"])
    trace.extend((2, i, item, position) for i, item in enumerate(k))
    v = _module(values, "self_attn.v_proj", n1, KV_HEADS * HEAD_DIM,
                values["model.layers.0.self_attn.v_proj.bias:"])
    trace.extend((3, i, item, position) for i, item in enumerate(v))
    rk = _rope(k, KV_HEADS, position)
    for head in range(KV_HEADS):
        for pair in range(32):
            base = head * HEAD_DIM
            trace.extend(((5, base + pair, rk[base + pair], position),
                          (5, base + pair + 32, rk[base + pair + 32], position)))
    cache_k.append(rk)
    cache_v.append(v)
    for head in range(KV_HEADS):
        for dimension in range(HEAD_DIM):
            index = head * HEAD_DIM + dimension
            trace.append((6, index, rk[index], position))
            trace.append((7, index, v[index], position))
    attention: list[int] = [0] * HIDDEN
    for head in range(HEADS):
        q_head = rq[head * HEAD_DIM:(head + 1) * HEAD_DIM]
        kv_head = head // 7
        scores = []
        for key_position, keys in enumerate(cache_k):
            score = attention_score(q_head, keys[kv_head * HEAD_DIM:(kv_head + 1) * HEAD_DIM],
                                    [True] * HEAD_DIM, position, key_position)
            if score.invalid or score.cache_miss:
                raise ArithmeticError("attention score invalid")
            scores.append(score)
            trace.append((8, key_position, score.score_f16, position))
        softmax = attention_softmax([x.score_f16 for x in scores], list(range(position + 1)),
                                    [x.causal for x in scores], [x.cache_miss for x in scores],
                                    [x.invalid for x in scores], position)
        if softmax.invalid or softmax.cache_miss or softmax.row_error:
            raise ArithmeticError("attention softmax invalid")
        for key_position, probability in enumerate(softmax.probabilities_f16):
            trace.append((9, key_position, probability, position))
        for dimension in range(HEAD_DIM):
            value = attention_value(
                list(softmax.probabilities_f16),
                [row[kv_head * HEAD_DIM + dimension] for row in cache_v],
                [True] * (position + 1), [False] * (position + 1))
            if value.invalid or value.cache_miss or value.row_error:
                raise ArithmeticError("attention value invalid")
            index = head * HEAD_DIM + dimension
            attention[index] = value.value_f16
            trace.append((10, index, value.value_f16, position))
    o = _module(values, "self_attn.o_proj", attention, HIDDEN)
    trace.extend((11, i, item, position) for i, item in enumerate(o))
    res1 = []
    for projected, residual in zip(o, activation, strict=True):
        value, invalid, _ = residual_add(projected, residual)
        if invalid:
            raise ArithmeticError("residual1 invalid")
        res1.append(value)
    trace.extend((12, i, item, position) for i, item in enumerate(res1))
    n2 = _expect_finite(rmsnorm(res1, values["model.layers.0.post_attention_layernorm.weight:"])[0], "norm2")
    trace.extend((13, i, item, position) for i, item in enumerate(n2))
    gate = _module(values, "mlp.gate_proj", n2, INTERMEDIATE)
    trace.extend((14, i, item, position) for i, item in enumerate(gate))
    up = _module(values, "mlp.up_proj", n2, INTERMEDIATE)
    trace.extend((15, i, item, position) for i, item in enumerate(up))
    silu = []
    for gate_item, up_item in zip(gate, up, strict=True):
        value, invalid, _ = (
            silu_gate_exp(gate_item, up_item)
            if accurate_silu else silu_gate(gate_item, up_item)
        )
        if invalid:
            raise ArithmeticError("SiLU invalid")
        silu.append(value)
    trace.extend((16, i, item, position) for i, item in enumerate(silu))
    down = _module(values, "mlp.down_proj", silu, HIDDEN)
    trace.extend((17, i, item, position) for i, item in enumerate(down))
    final = []
    for projected, residual in zip(down, res1, strict=True):
        value, invalid, _ = residual_add(projected, residual)
        if invalid:
            raise ArithmeticError("final residual invalid")
        final.append(value)
    trace.extend((18, i, item, position) for i, item in enumerate(final))
    return final, trace
