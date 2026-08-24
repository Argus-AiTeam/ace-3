#!/usr/bin/env python3
"""Bit-level Qwen2.5 half-split RoPE oracle for binary16 datapaths."""

from __future__ import annotations

import math
import struct

from awq_bit_oracle import q47_48_to_f16
from fp16_adaptation_oracle import decode_f16_q24, residual_add

HEAD_DIM = 64
ROTARY_PAIRS = HEAD_DIM // 2
ROPE_THETA = 1_000_000.0
MAX_POSITION = 32_768


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f16_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def qwen2_coefficient(position: int, pair_index: int) -> tuple[int, int]:
    if not 0 <= position < MAX_POSITION:
        raise ValueError("position outside authenticated checkpoint range")
    if not 0 <= pair_index < ROTARY_PAIRS:
        raise ValueError("rotary pair index outside head dimension")
    exponent = f32(f32(2.0 * pair_index) / f32(float(HEAD_DIM)))
    inv_frequency = f32(1.0 / f32(ROPE_THETA**exponent))
    angle = f32(f32(float(position)) * inv_frequency)
    return f16_bits(f32(math.cos(angle))), f16_bits(f32(math.sin(angle)))


def multiply_f16(a_bits: int, b_bits: int) -> tuple[int, bool, bool]:
    a_q24, a_finite, _, a_sign = decode_f16_q24(a_bits)
    b_q24, b_finite, _, b_sign = decode_f16_q24(b_bits)
    if not a_finite or not b_finite:
        return 0, True, False
    product = a_q24 * b_q24
    result, saturated = q47_48_to_f16(product)
    if product == 0:
        result = (int(a_sign ^ b_sign) << 15)
    return result, False, saturated


def negate_f16(value: int) -> int:
    return (value ^ 0x8000) & 0xFFFF


def rotate_pair(
    low_f16: int,
    high_f16: int,
    cos_f16: int,
    sin_f16: int,
) -> tuple[int, int, bool, bool]:
    low_cos, invalid_0, saturation_0 = multiply_f16(low_f16, cos_f16)
    high_sin, invalid_1, saturation_1 = multiply_f16(high_f16, sin_f16)
    high_cos, invalid_2, saturation_2 = multiply_f16(high_f16, cos_f16)
    low_sin, invalid_3, saturation_3 = multiply_f16(low_f16, sin_f16)
    invalid = invalid_0 or invalid_1 or invalid_2 or invalid_3
    if invalid:
        return 0, 0, True, False
    rotated_low, invalid_4, saturation_4 = residual_add(
        low_cos, negate_f16(high_sin)
    )
    rotated_high, invalid_5, saturation_5 = residual_add(high_cos, low_sin)
    return (
        rotated_low,
        rotated_high,
        invalid_4 or invalid_5,
        saturation_0
        or saturation_1
        or saturation_2
        or saturation_3
        or saturation_4
        or saturation_5,
    )


def self_test() -> None:
    cos_zero, sin_zero = qwen2_coefficient(0, 0)
    assert (cos_zero, sin_zero) == (0x3C00, 0x0000)
    assert rotate_pair(0x3C00, 0x4000, cos_zero, sin_zero) == (
        0x3C00,
        0x4000,
        False,
        False,
    )
    cos_value, sin_value = qwen2_coefficient(37, 31)
    assert (cos_value, sin_value) == (0x3C00, 0x03BC)
    assert rotate_pair(0x7C00, 0x3C00, cos_value, sin_value) == (
        0,
        0,
        True,
        False,
    )
    print(
        "QWEN2_ROPE_ORACLE_PASS head_dim=64 pairs=32 theta=1000000 "
        "half_split=pass fp16_mul_add=pass"
    )


if __name__ == "__main__":
    self_test()
