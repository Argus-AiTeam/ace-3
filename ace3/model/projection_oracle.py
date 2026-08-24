#!/usr/bin/env python3
"""Integer-only oracle for complete native-AWQ W4A16 projection outputs."""

from __future__ import annotations

from awq_bit_oracle import GROUP_SIZE, dot_group, q47_48_to_f16

GROUP_ACC_BITS = 96
MAX_GROUPS = 38
CROSS_ACC_BITS = GROUP_ACC_BITS + 6


def complete_projection_output(
    activations_f16: list[int],
    qweights_i32: list[int],
    qzeros_i32: list[int],
    scales_f16: list[int],
    logical_lane: int,
) -> tuple[int, int, bool, bool, list[int]]:
    if len(activations_f16) != len(qweights_i32):
        raise ValueError("activation and qweight lengths differ")
    if len(activations_f16) % GROUP_SIZE:
        raise ValueError("input feature count must be divisible by 128")
    group_count = len(activations_f16) // GROUP_SIZE
    if not 1 <= group_count <= MAX_GROUPS:
        raise ValueError(f"group count must be in [1, {MAX_GROUPS}]")
    if len(qzeros_i32) != group_count or len(scales_f16) != group_count:
        raise ValueError("one qzero word and scale are required per group")

    accumulator = 0
    invalid = False
    group_accumulators: list[int] = []
    for group_index in range(group_count):
        begin = group_index * GROUP_SIZE
        end = begin + GROUP_SIZE
        group_accumulator, group_invalid = dot_group(
            activations_f16[begin:end],
            qweights_i32[begin:end],
            qzeros_i32[group_index],
            scales_f16[group_index],
            logical_lane,
        )
        group_accumulators.append(group_accumulator)
        accumulator += group_accumulator
        invalid |= group_invalid

    if not -(1 << (CROSS_ACC_BITS - 1)) <= accumulator < (
        1 << (CROSS_ACC_BITS - 1)
    ):
        raise OverflowError("complete projection sum exceeded signed Q53.48")
    if invalid:
        return accumulator, 0x0000, True, False, group_accumulators
    result, saturated = q47_48_to_f16(accumulator)
    return accumulator, result, False, saturated, group_accumulators


def self_test() -> None:
    input_features = 7 * GROUP_SIZE
    activations = [0] * input_features
    qweights = [0] * input_features
    qzeros = [0] * 7
    scales = [0x3C00] * 7

    activations[0] = 0x0001
    qweights[0] = 0x00000001
    activations[GROUP_SIZE] = 0x8001
    qweights[GROUP_SIZE] = 0x00000001
    scales[1] = 0x3800
    accumulator, result, invalid, saturated, groups = complete_projection_output(
        activations, qweights, qzeros, scales, 0
    )
    assert groups[:2] == [1 << 24, -(1 << 23)]
    assert accumulator == 1 << 23
    assert (result, invalid, saturated) == (0x0000, False, False)
    assert q47_48_to_f16(1 << 24) == (0x0001, False)
    assert q47_48_to_f16(1 << 80) == (0x7BFF, True)
    assert q47_48_to_f16(-(1 << 80)) == (0xFBFF, True)
    assert 38 * ((1 << 95) - 1) <= (1 << 101) - 1
    assert -(38 * (1 << 95)) >= -(1 << 101)
    print(
        "PROJECTION_ORACLE_PASS checks=9 group_acc_bits=96 "
        "cross_acc_bits=102 max_groups=38 round_once=pass"
    )


if __name__ == "__main__":
    self_test()
