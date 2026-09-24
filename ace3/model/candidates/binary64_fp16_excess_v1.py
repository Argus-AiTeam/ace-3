"""Prospective scalar layer-output policy; no trajectory admission or live gate edits."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from ace3.model.awq_bit_oracle import f16_finite_parts

PROFILE_ID = "ace3-w4a16-layer-final-binary64-fp16-excess-v1"
PROFILE_VERSION = 1
REFERENCE_POLICY = "legacy-binary64-AWQ-fully-independent-propagation"
EXCESS_BUDGET = Fraction(1, 8)
ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATHS = (
    "ace3/contracts/candidates/binary64_fp16_excess_v1.json",
    "ace3/model/candidates/binary64_fp16_excess_v1.py",
    "ace3/model/awq_bit_oracle.py",
)


class InvalidOperand(ValueError):
    """The scalar does not satisfy the profile's input contract."""


class EvaluatorDefect(RuntimeError):
    """An internal numerical invariant failed; not an RTL rejection."""


def _finite_value(bits: int) -> Fraction:
    finite, sign, significand, exponent = f16_finite_parts(bits)
    if not finite:
        raise InvalidOperand("nonfinite_actual_fp16")
    if exponent >= 0:
        return Fraction(sign * significand * (1 << exponent))
    return Fraction(sign * significand, 1 << -exponent)


def _reference(text: str) -> float:
    if type(text) is not str:
        raise InvalidOperand("invalid_reference_encoding")
    try:
        value = float.fromhex(text)
    except ValueError as error:
        raise InvalidOperand("invalid_reference_encoding") from error
    except OverflowError as error:
        raise InvalidOperand("reference_binary64_overflow") from error
    if not math.isfinite(value):
        raise InvalidOperand("nonfinite_reference")
    if value.hex() != text:
        raise InvalidOperand("noncanonical_reference_encoding")
    if abs(value) > 65504:
        raise InvalidOperand("reference_fp16_range_overflow")
    return value


def _nearest(reference: float) -> tuple[int, Fraction]:
    exact = Fraction.from_float(reference)
    magnitude = abs(exact)
    low, high = 0, 0x7BFF
    while low < high:
        middle = (low + high + 1) // 2
        if _finite_value(middle) <= magnitude:
            low = middle
        else:
            high = middle - 1
    # Magnitude encodings are monotonic; the minimizer is one of these neighbors.
    nearest = min(
        (low, min(low + 1, 0x7BFF)),
        key=lambda bits: (abs(_finite_value(bits) - magnitude), bits & 1),
    )
    if math.copysign(1.0, reference) < 0:
        nearest |= 0x8000
    return nearest, abs(exact - _finite_value(nearest))


def evaluate_layer_final_output(
    *, actual_fp16_bits: int, reference_binary64_hex: str
) -> dict:
    if type(actual_fp16_bits) is not int or not 0 <= actual_fp16_bits <= 0xFFFF:
        raise InvalidOperand("invalid_actual_fp16_encoding")
    actual = _finite_value(actual_fp16_bits)
    reference = _reference(reference_binary64_hex)
    nearest, q = _nearest(reference)
    actual_error = abs(actual - Fraction.from_float(reference))
    excess_error = actual_error - q
    if excess_error < 0:
        raise EvaluatorDefect("negative_excess_error")
    return {
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "boundary": "layer-final-output",
        "precision": "W4A16",
        "reference_policy": REFERENCE_POLICY,
        "claim_scope": "scalar_output_error_only_not_trajectory_admission",
        "sources": [
            {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
            for path in SOURCE_PATHS
        ],
        "actual_fp16_bits": f"{actual_fp16_bits:04x}",
        "reference_binary64_hex": reference_binary64_hex,
        "nearest_fp16_bits": f"{nearest:04x}",
        "q": str(q),
        "actual_error": str(actual_error),
        "excess_error": str(excess_error),
        "excess_budget": str(EXCESS_BUDGET),
        "accepted": excess_error <= EXCESS_BUDGET,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-fp16-bits", required=True, help="exactly four hex digits")
    parser.add_argument("--reference-binary64-hex", required=True, help="canonical float.hex()")
    args = parser.parse_args()
    try:
        if len(args.actual_fp16_bits) != 4 or any(
            digit not in "0123456789abcdefABCDEF" for digit in args.actual_fp16_bits
        ):
            raise InvalidOperand("invalid_actual_fp16_encoding")
        result = evaluate_layer_final_output(
            actual_fp16_bits=int(args.actual_fp16_bits, 16),
            reference_binary64_hex=args.reference_binary64_hex,
        )
    except InvalidOperand as error:
        print(json.dumps({
            "profile_id": PROFILE_ID, "status": "FAIL", "reason": str(error),
            "actual_fp16_bits": args.actual_fp16_bits,
            "reference_binary64_hex": args.reference_binary64_hex,
        }, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
