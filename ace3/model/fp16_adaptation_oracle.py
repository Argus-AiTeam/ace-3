#!/usr/bin/env python3
"""Integer-only binary16 oracle for ACE-3 adaptation operators."""

from __future__ import annotations

import math

Q24_ONE = 1 << 24
Q24_HALF = 1 << 23
LN2_Q24 = 11_629_080
EPSILON_Q48 = 281_474_977
MAX_FINITE_F16 = 0x7BFF


def decode_f16_q24(bits: int) -> tuple[int, bool, bool, bool]:
    bits &= 0xFFFF
    sign = bool(bits & 0x8000)
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x3FF
    finite = exponent != 0x1F
    zero = exponent == 0 and fraction == 0
    if not finite:
        return 0, False, zero, sign
    magnitude = fraction if exponent == 0 else (0x400 | fraction) << (exponent - 1)
    return (-magnitude if sign else magnitude), True, zero, sign


def round_div_even_unsigned(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("unsigned RNE division domain violation")
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient & 1):
        quotient += 1
    return quotient


def round_shift_even_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value << -shift
    magnitude = abs(value)
    base = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and base & 1):
        base += 1
    return -base if value < 0 else base


def q24_to_f16(value: int, zero_sign: bool = False) -> tuple[int, bool]:
    sign = value < 0
    magnitude = abs(value)
    if magnitude == 0:
        return (0x8000 if zero_sign else 0), False
    most_significant_bit = magnitude.bit_length() - 1
    if most_significant_bit <= 9:
        return ((0x8000 if sign else 0) | magnitude), False

    shift = most_significant_bit - 10
    retained = magnitude >> shift
    if shift:
        remainder = magnitude & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if remainder > half or (remainder == half and retained & 1):
            retained += 1
    exponent = most_significant_bit - 24
    if retained == 0x800:
        retained >>= 1
        exponent += 1
    if exponent > 15:
        return ((0x8000 if sign else 0) | MAX_FINITE_F16), True
    return (
        (0x8000 if sign else 0)
        | ((exponent + 15) << 10)
        | (retained & 0x3FF)
    ), False


def residual_add(a_f16: int, b_f16: int) -> tuple[int, bool, bool]:
    a, a_finite, a_zero, a_sign = decode_f16_q24(a_f16)
    b, b_finite, b_zero, b_sign = decode_f16_q24(b_f16)
    if not a_finite or not b_finite:
        return 0, True, False
    result, saturated = q24_to_f16(
        a + b, zero_sign=a_zero and b_zero and a_sign and b_sign
    )
    return result, False, saturated


def sigmoid_q24(gate_q24: int) -> int:
    magnitude = abs(gate_q24)
    term = round_div_even_unsigned(
        magnitude << 24, 2 * (Q24_ONE + magnitude)
    )
    return Q24_HALF - term if gate_q24 < 0 else Q24_HALF + term


def silu_gate(gate_f16: int, up_f16: int) -> tuple[int, bool, bool]:
    gate, gate_finite, _, gate_sign = decode_f16_q24(gate_f16)
    up, up_finite, _, up_sign = decode_f16_q24(up_f16)
    if not gate_finite or not up_finite:
        return 0, True, False
    product_q72 = gate * up * sigmoid_q24(gate)
    result_q24 = round_shift_even_signed(product_q72, 48)
    result, saturated = q24_to_f16(
        result_q24, zero_sign=gate_sign ^ up_sign
    )
    return result, False, saturated


def _exp_negative_q24(magnitude_q24: int) -> int:
    quotient, remainder = divmod(magnitude_q24, LN2_Q24)
    coefficients = (
        Q24_ONE,
        -Q24_ONE,
        Q24_ONE // 2,
        -2_796_203,
        699_051,
        -139_810,
        23_302,
        -3_329,
    )
    polynomial = coefficients[-1]
    for coefficient in reversed(coefficients[1:-1]):
        polynomial = coefficient + round_shift_even_signed(
            remainder * polynomial, 24
        )
    exponential = coefficients[0] + round_shift_even_signed(
        remainder * polynomial, 24
    )
    if quotient >= 63:
        return 0
    return round_shift_even_signed(exponential, quotient)


def silu_gate_exp(gate_f16: int, up_f16: int) -> tuple[int, bool, bool]:
    gate, gate_finite, _, gate_sign = decode_f16_q24(gate_f16)
    up, up_finite, _, up_sign = decode_f16_q24(up_f16)
    if not gate_finite or not up_finite:
        return 0, True, False
    exponential = _exp_negative_q24(abs(gate))
    negative_sigmoid = round_div_even_unsigned(
        exponential << 24, Q24_ONE + exponential
    )
    sigmoid = negative_sigmoid if gate < 0 else Q24_ONE - negative_sigmoid
    result_q24 = round_shift_even_signed(gate * up * sigmoid, 48)
    result, saturated = q24_to_f16(
        result_q24, zero_sign=gate_sign ^ up_sign
    )
    return result, False, saturated


def rmsnorm(
    activations_f16: list[int],
    weights_f16: list[int],
    epsilon_q48: int = EPSILON_Q48,
) -> tuple[list[tuple[int, bool, bool]], int, int]:
    if not activations_f16 or len(activations_f16) != len(weights_f16):
        raise ValueError("RMSNorm activation and weight vectors must match")
    decoded_activations = [decode_f16_q24(value) for value in activations_f16]
    decoded_weights = [decode_f16_q24(value) for value in weights_f16]
    invalid = any(not item[1] for item in decoded_activations + decoded_weights)
    sumsq = sum(item[0] * item[0] for item in decoded_activations)
    mean_q48 = round_div_even_unsigned(
        sumsq + epsilon_q48 * len(activations_f16),
        len(activations_f16),
    )
    rms_q24 = math.isqrt(mean_q48)
    if invalid:
        return [(0, True, False)] * len(activations_f16), mean_q48, rms_q24
    if rms_q24 == 0:
        raise ArithmeticError("positive epsilon produced zero RMS")

    outputs: list[tuple[int, bool, bool]] = []
    for activation, weight in zip(
        decoded_activations, decoded_weights, strict=True
    ):
        product = activation[0] * weight[0]
        magnitude = round_div_even_unsigned(abs(product), rms_q24)
        result_q24 = -magnitude if product < 0 else magnitude
        result, saturated = q24_to_f16(
            result_q24, zero_sign=activation[3] ^ weight[3]
        )
        outputs.append((result, False, saturated))
    return outputs, mean_q48, rms_q24


def self_test() -> None:
    assert decode_f16_q24(0x3C00)[0] == Q24_ONE
    assert decode_f16_q24(0x0001)[0] == 1
    assert q24_to_f16(Q24_ONE) == (0x3C00, False)
    assert residual_add(0x3C00, 0x3C00) == (0x4000, False, False)
    assert silu_gate(0x3C00, 0x3C00) == (0x3A00, False, False)
    outputs, _, rms = rmsnorm([0x3C00, 0xBC00], [0x3C00, 0x3C00])
    assert [item[0] for item in outputs] == [0x3C00, 0xBC00]
    assert rms > 0
    print(
        "FP16_ADAPTATION_ORACLE_PASS checks=7 q_fraction=24 "
        "epsilon_q48=281474977 sigmoid=rational_rne"
    )


if __name__ == "__main__":
    self_test()
