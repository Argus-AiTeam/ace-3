#!/usr/bin/env python3
"""Accepted W4A16 projection arithmetic at an FP16 stage boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from awq_bit_oracle import AWQ_REVERSE_ORDER, GROUP_SIZE


def project_fp16_stage(
    activations_f16: Sequence[int],
    qweight_i32: Sequence[int],
    qzeros_i32: Sequence[int],
    scales_f16: Sequence[int],
    out_features: int,
    bias_f16: Sequence[int] | None = None,
) -> np.ndarray:
    input_features = len(activations_f16)
    if input_features == 0 or input_features % GROUP_SIZE:
        raise ValueError("input feature count must be a nonzero multiple of 128")
    if out_features == 0 or out_features % 8:
        raise ValueError("output feature count must be a nonzero multiple of 8")
    groups = input_features // GROUP_SIZE
    packed_columns = out_features // 8
    if len(qweight_i32) != input_features * packed_columns:
        raise ValueError("qweight geometry mismatch")
    if len(qzeros_i32) != groups * packed_columns:
        raise ValueError("qzeros geometry mismatch")
    if len(scales_f16) != groups * out_features:
        raise ValueError("scale geometry mismatch")
    if bias_f16 is not None and len(bias_f16) != out_features:
        raise ValueError("bias geometry mismatch")

    qweight = np.asarray(qweight_i32, dtype="<u4").reshape(
        input_features, packed_columns
    )
    qzeros = np.asarray(qzeros_i32, dtype="<u4").reshape(
        groups, packed_columns
    )
    scales = np.asarray(scales_f16, dtype="<u2").view("<f2").astype(
        np.float64
    ).reshape(groups, out_features)
    quantized = np.empty((input_features, out_features), dtype=np.float64)
    zeros = np.empty((groups, out_features), dtype=np.float64)
    for channel in range(out_features):
        packed, lane = divmod(channel, 8)
        shift = 4 * AWQ_REVERSE_ORDER[lane]
        quantized[:, channel] = (qweight[:, packed] >> shift) & 0xF
        zeros[:, channel] = (qzeros[:, packed] >> shift) & 0xF
    weight = (
        quantized - np.repeat(zeros, GROUP_SIZE, axis=0)
    ) * np.repeat(scales, GROUP_SIZE, axis=0)
    activation = torch.from_numpy(
        np.asarray(activations_f16, dtype="<u2")
        .view("<f2")
        .astype(np.float64)
    )
    output = activation @ torch.from_numpy(weight)
    if bias_f16 is not None:
        bias = (
            np.asarray(bias_f16, dtype="<u2")
            .view("<f2")
            .astype(np.float64)
        )
        output = output + torch.from_numpy(bias)
    return np.asarray(output.numpy(), dtype="<f2").view("<u2")


def project_named_module(
    values: Mapping[str, Sequence[int]],
    prefix: str,
    activations_f16: Sequence[int],
    out_features: int,
    bias_f16: Sequence[int] | None = None,
) -> np.ndarray:
    namespace = f"model.layers.0.{prefix}"
    return project_fp16_stage(
        activations_f16,
        values[f"{namespace}.qweight:"],
        values[f"{namespace}.qzeros:"],
        values[f"{namespace}.scales:"],
        out_features,
        bias_f16,
    )
