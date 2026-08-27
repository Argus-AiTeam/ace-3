#!/usr/bin/env python3
"""Generate authenticated standalone final-RMSNorm vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

HIDDEN_SIZE = 896
EPSILON_Q48 = 281474977
OFFICIAL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
OFFICIAL_SHA256 = "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f16_to_q24(bits: int) -> tuple[int, bool, int]:
    sign = (bits >> 15) & 1
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x3FF
    if exponent == 0x1F:
        return 0, False, sign
    magnitude = fraction if exponent == 0 else (0x400 | fraction) << (exponent - 1)
    return (-magnitude if sign else magnitude), True, sign


def round_div_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    return quotient + int(doubled > denominator or (doubled == denominator and quotient & 1))


def q24_to_f16(value: int, zero_sign: int) -> tuple[int, bool]:
    sign = int(value < 0)
    magnitude = abs(value)
    if magnitude == 0:
        return zero_sign << 15, False
    msb = magnitude.bit_length() - 1
    if msb < 10:
        return (sign << 15) | magnitude, False
    shift = msb - 10
    retained = magnitude >> shift
    if shift:
        remainder = magnitude & ((1 << shift) - 1)
        halfway = 1 << (shift - 1)
        retained += int(remainder > halfway or (remainder == halfway and retained & 1))
    exponent = msb - 24
    if retained == 0x800:
        retained >>= 1
        exponent += 1
    if exponent > 15:
        return (sign << 15) | 0x7BFF, True
    encoded_exponent = exponent + 15
    return (sign << 15) | (encoded_exponent << 10) | (retained & 0x3FF), False


def rmsnorm_case(activations: list[int], weights: list[int]) -> list[int]:
    decoded_activations: list[int] = []
    decoded_weights: list[int] = []
    zero_signs: list[int] = []
    valid = True
    sum_squares = 0
    for activation_bits, weight_bits in zip(activations, weights, strict=True):
        activation_q24, activation_finite, activation_sign = f16_to_q24(activation_bits)
        weight_q24, weight_finite, weight_sign = f16_to_q24(weight_bits)
        decoded_activations.append(activation_q24)
        decoded_weights.append(weight_q24)
        zero_signs.append(activation_sign ^ weight_sign)
        valid &= activation_finite and weight_finite
        sum_squares += activation_q24 * activation_q24
    mean_q48 = round_div_even(sum_squares + EPSILON_Q48 * HIDDEN_SIZE, HIDDEN_SIZE)
    root_q24 = math.isqrt(mean_q48)
    outputs: list[int] = []
    for activation_q24, weight_q24, zero_sign in zip(
        decoded_activations, decoded_weights, zero_signs, strict=True
    ):
        if not valid:
            outputs.append(0)
            continue
        product = activation_q24 * weight_q24
        magnitude = round_div_even(abs(product), root_q24)
        output_q24 = -magnitude if product < 0 else magnitude
        outputs.append(q24_to_f16(output_q24, zero_sign)[0])
    return outputs


def activation_cases() -> list[list[int]]:
    patterns = [
        [0x0000],
        [0x3C00, 0xBC00],
        [0x3400, 0xB800, 0x4000, 0xB400, 0x3A00, 0xC100, 0x3800],
        [0x0001, 0x8001, 0x0000, 0x8000, 0x2C00, 0xAC00, 0x3E00, 0xBE00],
    ]
    return [[pattern[index % len(pattern)] for index in range(HIDDEN_SIZE)] for pattern in patterns]


def write_memh(path: Path, values: list[int]) -> None:
    path.write_text("".join(f"{value:04x}\n" for value in values), encoding="ascii")


def generate(checkpoint: Path, output_dir: Path, expected_sha256: str) -> None:
    if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256):
        raise ValueError("expected SHA-256 must be 64 lowercase hexadecimal characters")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is not a regular file: {checkpoint}")
    before = checkpoint.stat()
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {actual_sha256}")
    with safe_open(checkpoint, framework="np", device="cpu") as tensors:
        if "model.norm.weight" not in tensors.keys():
            raise ValueError("checkpoint lacks model.norm.weight")
        weight = tensors.get_tensor("model.norm.weight")
    if weight.shape != (HIDDEN_SIZE,) or weight.dtype != np.float16:
        raise ValueError(f"model.norm.weight must be float16[{HIDDEN_SIZE}]")
    weights = [int(value) for value in np.asarray(weight).view(np.uint16)]
    cases = activation_cases()
    expected = [value for case in cases for value in rmsnorm_case(case, weights)]
    after = checkpoint.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_mode) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    ):
        raise RuntimeError("checkpoint metadata changed during read-only generation")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_memh(output_dir / "activations.memh", [value for case in cases for value in case])
    write_memh(output_dir / "weights.memh", weights)
    write_memh(output_dir / "expected.memh", expected)
    manifest = {
        "schema": "ace3-final-rmsnorm-vectors-v1",
        "official_revision": OFFICIAL_REVISION,
        "checkpoint_sha256": actual_sha256,
        "hidden_size": HIDDEN_SIZE,
        "case_count": len(cases),
        "expected_comparisons": len(expected),
        "files": {
            "activations": "activations.memh",
            "weights": "weights.memh",
            "expected": "expected.memh",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=OFFICIAL_SHA256)
    args = parser.parse_args()
    generate(args.checkpoint, args.output_dir, args.expected_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
