#!/usr/bin/env python3
"""Integer-only bit oracle for the ACE-3 MP AWQ G128 dot lane."""

from __future__ import annotations

AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
GROUP_SIZE = 128
ACC_FRAC_BITS = 48
ACC_BITS = 96


def awq_nibble(word: int, logical_lane: int) -> int:
    if not 0 <= logical_lane < 8:
        raise ValueError("logical lane must be in [0, 7]")
    return (word >> (4 * AWQ_REVERSE_ORDER[logical_lane])) & 0xF


def f16_finite_parts(bits: int) -> tuple[bool, int, int, int]:
    bits &= 0xFFFF
    sign = -1 if bits & 0x8000 else 1
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x3FF
    if exponent == 0x1F:
        return False, sign, 0, 0
    if exponent == 0:
        return True, sign, fraction, -24
    return True, sign, 0x400 | fraction, exponent - 25


def product_q47_48(
    activation_f16: int, scale_f16: int, quantized_delta: int
) -> tuple[int, bool]:
    act_finite, act_sign, act_sig, act_exp = f16_finite_parts(activation_f16)
    scale_finite, scale_sign, scale_sig, scale_exp = f16_finite_parts(scale_f16)
    if not act_finite or not scale_finite:
        return 0, True
    magnitude = act_sig * scale_sig * abs(quantized_delta)
    shift = act_exp + scale_exp + ACC_FRAC_BITS
    if shift < 0:
        raise AssertionError("binary16 product is finer than Q47.48")
    value = magnitude << shift
    if act_sign * scale_sign * quantized_delta < 0:
        value = -value
    return value, False


def dot_group(
    activations_f16: list[int],
    qweights_i32: list[int],
    qzeros_i32: int,
    scale_f16: int,
    logical_lane: int,
) -> tuple[int, bool]:
    if len(activations_f16) != GROUP_SIZE or len(qweights_i32) != GROUP_SIZE:
        raise ValueError("one G128 group requires exactly 128 pairs")
    qzero = awq_nibble(qzeros_i32, logical_lane)
    accumulator = 0
    invalid = False
    for activation, qweight in zip(activations_f16, qweights_i32, strict=True):
        delta = awq_nibble(qweight, logical_lane) - qzero
        product, product_invalid = product_q47_48(activation, scale_f16, delta)
        accumulator += product
        invalid |= product_invalid
    if not -(1 << 95) <= accumulator < (1 << 95):
        raise OverflowError("contract-valid G128 sum exceeded signed Q47.48")
    return accumulator, invalid


def _round_shift_rne(value: int, shift: int) -> int:
    if shift <= 0:
        return value << -shift
    retained = value >> shift
    remainder = value & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    return retained + (remainder > halfway or (remainder == halfway and retained & 1))


def q47_48_to_f16(accumulator: int) -> tuple[int, bool]:
    if accumulator == 0:
        return 0x0000, False
    sign_bit = 0x8000 if accumulator < 0 else 0
    magnitude = abs(accumulator)
    msb = magnitude.bit_length() - 1
    if msb < 34:
        fraction = _round_shift_rne(magnitude, 24)
        if fraction == 0:
            return 0x0000, False
        if fraction >= 1024:
            return sign_bit | 0x0400, False
        return sign_bit | fraction, False

    exponent = msb - ACC_FRAC_BITS
    significand = _round_shift_rne(magnitude, msb - 10)
    if significand == 2048:
        significand = 1024
        exponent += 1
    if exponent > 15:
        return sign_bit | 0x7BFF, True
    return sign_bit | ((exponent + 15) << 10) | (significand & 0x3FF), False


def complete_dot(
    activations_f16: list[int],
    qweights_i32: list[int],
    qzeros_i32: int,
    scale_f16: int,
    logical_lane: int,
) -> tuple[int, int, bool, bool]:
    accumulator, invalid = dot_group(
        activations_f16,
        qweights_i32,
        qzeros_i32,
        scale_f16,
        logical_lane,
    )
    if invalid:
        return accumulator, 0x0000, True, False
    result, saturated = q47_48_to_f16(accumulator)
    return accumulator, result, False, saturated


def self_test() -> None:
    assert [awq_nibble(0x76543210, lane) for lane in range(8)] == [
        0,
        4,
        1,
        5,
        2,
        6,
        3,
        7,
    ]
    assert product_q47_48(0x3C00, 0x3C00, 1) == (1 << 48, False)
    assert product_q47_48(0x0001, 0x0001, 1) == (1, False)
    assert q47_48_to_f16(1 << 24) == (0x0001, False)
    assert q47_48_to_f16(3 << 23) == (0x0002, False)
    assert q47_48_to_f16(-(1 << 23)) == (0x0000, False)
    assert q47_48_to_f16((1 << 80)) == (0x7BFF, True)
    print("ORACLE_PASS integer-only FP16/Q47.48 checks=7")


if __name__ == "__main__":
    self_test()
