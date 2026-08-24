#!/usr/bin/env python3
"""Independent bit-level oracle for the ACE-3 FP16 attention block."""

from __future__ import annotations

import math
from dataclasses import dataclass

from fp16_adaptation_oracle import (
    decode_f16_q24,
    q24_to_f16,
    round_div_even_unsigned,
    round_shift_even_signed,
)

QUERY_HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64
LOG2_E_Q16 = 94548
EXP_LUT_Q24 = (
    16777216,
    16065917,
    15384775,
    14732511,
    14107901,
    13509772,
    12937002,
    12388516,
    11863283,
    11360319,
    10878679,
    10417458,
    9975792,
    9552851,
    9147842,
    8760003,
    8388608,
)


@dataclass(frozen=True)
class ScoreResult:
    score_f16: int
    causal: bool
    cache_miss: bool
    invalid: bool
    saturation: bool


@dataclass(frozen=True)
class SoftmaxResult:
    probabilities_f16: tuple[int, ...]
    row_error: bool
    cache_miss: bool
    invalid: bool


@dataclass(frozen=True)
class ValueResult:
    value_f16: int
    row_error: bool
    cache_miss: bool
    invalid: bool
    saturation: bool


def mapped_kv_head(query_head: int) -> int:
    if not 0 <= query_head < QUERY_HEADS:
        raise ValueError("query head outside Qwen2.5 geometry")
    return query_head // 7


def attention_score(
    q_f16: list[int],
    k_f16: list[int],
    cache_hits: list[bool],
    query_position: int,
    key_position: int,
) -> ScoreResult:
    if not (
        len(q_f16) == len(k_f16) == len(cache_hits) == HEAD_DIM
    ):
        raise ValueError("score requires exactly 64 Q/K/cache records")
    accumulator_q48 = 0
    invalid = False
    for q_bits, k_bits in zip(q_f16, k_f16, strict=True):
        q_q24, q_finite, _, _ = decode_f16_q24(q_bits)
        k_q24, k_finite, _, _ = decode_f16_q24(k_bits)
        invalid |= not q_finite or not k_finite
        accumulator_q48 += q_q24 * k_q24
    causal = key_position <= query_position
    cache_miss = causal and not all(cache_hits)
    scaled_q24 = round_shift_even_signed(accumulator_q48, 27)
    score_f16, saturation = q24_to_f16(scaled_q24)
    if not causal or cache_miss or invalid:
        return ScoreResult(0, causal, cache_miss, invalid, False)
    return ScoreResult(score_f16, causal, False, False, saturation)


def exp_approx_q24(delta_q24: int) -> int:
    if delta_q24 < 0:
        raise ValueError("max-subtracted delta must be nonnegative")
    y_q16 = (delta_q24 * LOG2_E_Q16) >> 24
    integer_part = y_q16 >> 16
    if integer_part >= 25:
        return 0
    table_index = (y_q16 >> 12) & 0xF
    fraction = y_q16 & 0xFFF
    upper = EXP_LUT_Q24[table_index]
    lower = EXP_LUT_Q24[table_index + 1]
    interpolation_drop = round_div_even_unsigned(
        (upper - lower) * fraction, 1 << 12
    )
    interpolated = upper - interpolation_drop
    return round_div_even_unsigned(interpolated, 1 << integer_part)


def attention_softmax(
    scores_f16: list[int],
    key_positions: list[int],
    causal_flags: list[bool],
    cache_misses: list[bool],
    invalid_flags: list[bool],
    query_position: int,
) -> SoftmaxResult:
    count = len(scores_f16)
    if not (
        count
        and len(key_positions)
        == len(causal_flags)
        == len(cache_misses)
        == len(invalid_flags)
        == count
    ):
        raise ValueError("softmax row lengths disagree")
    decoded: list[int] = []
    expected_causal: list[bool] = []
    invalid = False
    cache_miss = False
    for score, key_position, causal, miss, upstream_invalid in zip(
        scores_f16,
        key_positions,
        causal_flags,
        cache_misses,
        invalid_flags,
        strict=True,
    ):
        score_q24, finite, _, _ = decode_f16_q24(score)
        decoded.append(score_q24)
        eligible = key_position <= query_position
        expected_causal.append(eligible)
        invalid |= causal != eligible or upstream_invalid or not finite
        cache_miss |= eligible and miss
    eligible_scores = [
        score for score, eligible in zip(decoded, expected_causal, strict=True)
        if eligible
    ]
    if invalid or cache_miss or not eligible_scores:
        return SoftmaxResult(
            (0,) * count,
            True,
            cache_miss,
            invalid or not eligible_scores,
        )
    maximum = max(eligible_scores)
    exponentials = [
        exp_approx_q24(maximum - score) if eligible else 0
        for score, eligible in zip(decoded, expected_causal, strict=True)
    ]
    denominator = sum(exponentials)
    if denominator == 0:
        return SoftmaxResult((0,) * count, True, False, True)
    probabilities = tuple(
        q24_to_f16(
            round_div_even_unsigned(exponential << 24, denominator)
        )[0]
        for exponential in exponentials
    )
    return SoftmaxResult(probabilities, False, False, False)


def attention_value(
    probabilities_f16: list[int],
    values_f16: list[int],
    value_hits: list[bool],
    row_errors: list[bool],
) -> ValueResult:
    count = len(probabilities_f16)
    if not (
        count
        and len(values_f16)
        == len(value_hits)
        == len(row_errors)
        == count
    ):
        raise ValueError("value-composition row lengths disagree")
    accumulator_q48 = 0
    invalid = False
    cache_miss = False
    row_error = any(row_errors)
    for probability, value, hit in zip(
        probabilities_f16, values_f16, value_hits, strict=True
    ):
        probability_q24, probability_finite, _, probability_sign = (
            decode_f16_q24(probability)
        )
        value_q24, value_finite, _, _ = decode_f16_q24(value)
        nonzero = (probability & 0x7FFF) != 0
        invalid |= (
            not probability_finite
            or (nonzero and probability_sign)
            or (nonzero and not value_finite)
        )
        cache_miss |= nonzero and not hit
        accumulator_q48 += probability_q24 * value_q24
    composed_q24 = round_shift_even_signed(accumulator_q48, 24)
    value_f16, saturation = q24_to_f16(composed_q24)
    if row_error or cache_miss or invalid:
        return ValueResult(0, True, cache_miss, invalid, False)
    return ValueResult(value_f16, False, False, False, saturation)


def self_test() -> None:
    score = attention_score(
        [0x3C00] * HEAD_DIM,
        [0x3C00] * HEAD_DIM,
        [True] * HEAD_DIM,
        0,
        0,
    )
    assert score == ScoreResult(0x4800, True, False, False, False)
    ties = attention_softmax(
        [0x3C00] * 4,
        [0, 1, 2, 3],
        [True] * 4,
        [False] * 4,
        [False] * 4,
        3,
    )
    assert ties == SoftmaxResult(
        (0x3400, 0x3400, 0x3400, 0x3400),
        False,
        False,
        False,
    )
    masked = attention_softmax(
        [0x0000, 0x7BFF],
        [0, 1],
        [True, False],
        [False, False],
        [False, False],
        0,
    )
    assert masked.probabilities_f16 == (0x3C00, 0x0000)
    value = attention_value(
        [0x3800, 0x3800],
        [0x4000, 0x4400],
        [True, True],
        [False, False],
    )
    assert value == ValueResult(0x4200, False, False, False, False)
    for step in range(0, 16 * 256 + 1):
        delta = step / 256.0
        approximate = exp_approx_q24(
            round(delta * (1 << 24))
        ) / float(1 << 24)
        exact = math.exp(-delta)
        assert abs(approximate - exact) < (
            exact * 0.0003 + 1.0 / (1 << 24)
        )
    print(
        "ACE3_ATTENTION_ORACLE_PASS score=fp16_q48_scale8 "
        "softmax=q24_lut16_interp_rne value=fp16_q48 "
        "gqa=14_to_2 causal=pass approximation_bound=pass"
    )


if __name__ == "__main__":
    self_test()
