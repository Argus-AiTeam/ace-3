"""General frozen cases for the prospective policy, not a retained-trajectory run."""

from __future__ import annotations

import ast
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import binary64_fp16_excess_v1 as candidate

LEGACY_SOURCE = Path(
    "/home/argustest/argustest2/ace3-continuous-generation-20260906/"
    "ace3/model/continuous_kv_rtl_backend.py"
)
WITNESS_REFERENCES = (
    ("layer03-position0-index62", "0x1.83c82dc02fd03p+10"),
    ("layer04-position0-index62", "0x1.8428911196fcfp+10"),
    ("layer06-position0-index62", "0x1.8aa3e286b789cp+10"),
    ("layer07-position0-index62", "0x1.8ab7c1503ecc5p+10"),
    ("layer08-position0-index62", "0x1.8ae0295917155p+10"),
)


def half(bits: int) -> float:
    return struct.unpack(">e", bits.to_bytes(2, "big"))[0]


def nearest_word(reference: float) -> int:
    return int.from_bytes(struct.pack(">e", reference), "big")


def general_cases() -> list[dict]:
    gaps = {0, 1, 2, 3, 0x1FF, 0x200, 0x3FD, 0x3FE, 0x3FF, 0x7BFE}
    for exponent in range(1, 31):
        base = exponent << 10
        gaps.update(base + offset for offset in (-1, 0, 1, 511, 512, 1022))
    gaps.update(random.Random(0xACE30001).sample(range(0x7BFF), 128))
    cases = []
    for low in sorted(gaps):
        a, b = half(low), half(low + 1)
        for quarter in range(5):
            magnitude = a + (b - a) * quarter / 4
            for sign in (0, 0x8000):
                reference = math.copysign(magnitude, -1.0 if sign else 1.0)
                for word in (low, low + 1):
                    cases.append({
                        "id": f"gap-{low:04x}-quarter{quarter}-sign{sign:04x}-a{word:04x}",
                        "actual_bits": word | sign,
                        "reference_hex": reference.hex(),
                    })
    for sign in (0, 0x8000):
        for label, reference, expected in (
            ("below", math.nextafter(1024.4375, -math.inf), False),
            ("inclusive", 1024.4375, True),
            ("above", math.nextafter(1024.4375, math.inf), True),
        ):
            cases.append({
                "id": f"budget-{label}-sign{sign:04x}",
                "actual_bits": 0x6401 | sign,
                "reference_hex": math.copysign(reference, -1.0 if sign else 1.0).hex(),
                "expected_accepted": expected,
            })
    for label, actual, reference in (
        ("positive-zero", 0x0000, 0.0),
        ("negative-zero", 0x8000, -0.0),
        ("opposite-zero", 0x0000, -0.0),
        ("tiny-positive", 0x0000, math.ulp(0.0)),
        ("tiny-negative", 0x8000, -math.ulp(0.0)),
        ("tiny-budget-subtraction", 0x3000, math.ulp(0.0)),
        ("tiny-budget-cancellation", 0x3000, -math.ulp(0.0)),
        ("positive-limit", 0x7BFF, 65504.0),
        ("negative-limit", 0xFBFF, -65504.0),
        ("opposite-limits", 0x7BFF, -65504.0),
        ("just-inside-limit", 0x7BFF, math.nextafter(65504.0, 0.0)),
    ):
        cases.append({"id": label, "actual_bits": actual, "reference_hex": reference.hex()})
    return cases


def legacy_comparison():
    # Execute the original scalar-array function without importing its RTL runtime.
    source = ast.parse(LEGACY_SOURCE.read_text())
    selected = []
    for node in source.body:
        if isinstance(node, ast.FunctionDef) and node.name == "float_comparison":
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in ("FLOAT_POLICY", "FLOAT_TOLERANCE")
            for target in node.targets
        ):
            selected.append(node)
    if len(selected) != 3:
        raise RuntimeError("legacy comparison source contract changed")

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    namespace = {"np": np, "require": require}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(LEGACY_SOURCE), "exec"), namespace)
    return namespace["float_comparison"]


class Binary64Fp16ExcessTests(unittest.TestCase):
    def test_general_cases_against_independent_ieee_conversion(self):
        cases = general_cases()
        self.assertEqual(len(cases), len({case["id"] for case in cases}))
        for case in cases:
            with self.subTest(case=case["id"]):
                reference = float.fromhex(case["reference_hex"])
                nearest = nearest_word(reference)
                exact = Fraction.from_float(reference)
                q = abs(Fraction.from_float(half(nearest)) - exact)
                actual_error = abs(Fraction.from_float(half(case["actual_bits"])) - exact)
                result = candidate.evaluate_layer_final_output(
                    actual_fp16_bits=case["actual_bits"],
                    reference_binary64_hex=case["reference_hex"],
                )
                self.assertEqual(result["nearest_fp16_bits"], f"{nearest:04x}")
                self.assertEqual(Fraction(result["q"]), q)
                self.assertEqual(Fraction(result["actual_error"]), actual_error)
                self.assertEqual(Fraction(result["excess_error"]), actual_error - q)
                self.assertGreaterEqual(Fraction(result["excess_error"]), 0)
                self.assertEqual(result["accepted"], actual_error - q <= Fraction(1, 8))
                if "expected_accepted" in case:
                    self.assertEqual(result["accepted"], case["expected_accepted"])

    def test_all_finite_binary16_decode_patterns(self):
        for bits in range(0x10000):
            if bits & 0x7C00 != 0x7C00:
                self.assertEqual(candidate._finite_value(bits), Fraction.from_float(half(bits)))

    def test_tiny_reference_arithmetic_is_not_binary64_subtraction(self):
        tiny = Fraction.from_float(math.ulp(0.0))
        result = candidate.evaluate_layer_final_output(
            actual_fp16_bits=0x3000, reference_binary64_hex=math.ulp(0.0).hex()
        )
        self.assertEqual(Fraction(result["actual_error"]), Fraction(1, 8) - tiny)
        self.assertEqual(Fraction(result["excess_error"]), Fraction(1, 8) - 2 * tiny)

    def test_five_reference_only_lower_bound_regressions(self):
        for identity, reference_hex in WITNESS_REFERENCES:
            with self.subTest(case=identity):
                reference = float.fromhex(reference_hex)
                nearest = nearest_word(reference)
                adjacent = [
                    Fraction.from_float(half(bits))
                    for bits in (nearest - 1, nearest, nearest + 1)
                ]
                lower_bound = min(abs(value - Fraction.from_float(reference)) for value in adjacent)
                self.assertGreater(lower_bound, Fraction(1, 8))
                result = candidate.evaluate_layer_final_output(
                    actual_fp16_bits=nearest, reference_binary64_hex=reference_hex
                )
                self.assertEqual(Fraction(result["q"]), lower_bound)
                self.assertEqual(result["excess_error"], "0")
                self.assertTrue(result["accepted"])

    def test_invalid_fp16_and_nonfinite_patterns_fail_explicitly(self):
        for bits in (-1, 65536, True, False, "0000", 0.0, None):
            with self.subTest(bits=bits), self.assertRaisesRegex(
                candidate.InvalidOperand, "invalid_actual_fp16_encoding"
            ):
                candidate.evaluate_layer_final_output(
                    actual_fp16_bits=bits, reference_binary64_hex=0.0.hex()
                )
        for bits in (0x7C00, 0xFC00, 0x7C01, 0xFC01, 0x7E00, 0xFE00, 0x7FFF, 0xFFFF):
            with self.subTest(bits=bits), self.assertRaisesRegex(
                candidate.InvalidOperand, "nonfinite_actual_fp16"
            ):
                candidate.evaluate_layer_final_output(
                    actual_fp16_bits=bits, reference_binary64_hex=0.0.hex()
                )

    def test_invalid_reference_nonfinite_and_overflow_fail(self):
        for text, reason in (
            ("nan", "nonfinite_reference"), ("inf", "nonfinite_reference"),
            ("-inf", "nonfinite_reference"), ("not-hex", "invalid_reference_encoding"),
            ("1.0", "noncanonical_reference_encoding"), (0.0, "invalid_reference_encoding"),
            ("0x1p-9999", "noncanonical_reference_encoding"),
            ("0x1p+9999", "reference_binary64_overflow"),
            (math.nextafter(65504.0, math.inf).hex(), "reference_fp16_range_overflow"),
            (math.nextafter(-65504.0, -math.inf).hex(), "reference_fp16_range_overflow"),
            (65520.0.hex(), "reference_fp16_range_overflow"),
            ((-65520.0).hex(), "reference_fp16_range_overflow"),
        ):
            with self.subTest(reference=text), self.assertRaisesRegex(candidate.InvalidOperand, reason):
                candidate.evaluate_layer_final_output(actual_fp16_bits=0, reference_binary64_hex=text)

    def test_negative_excess_is_an_evaluator_defect_not_clamped(self):
        with patch.object(candidate, "_nearest", return_value=(0, Fraction(1))):
            with self.assertRaisesRegex(candidate.EvaluatorDefect, "negative_excess_error"):
                candidate.evaluate_layer_final_output(
                    actual_fp16_bits=0, reference_binary64_hex=0.0.hex()
                )

    def test_actual_archived_legacy_gate_is_unchanged(self):
        compare = legacy_comparison()
        for actual, reference, expected in (
            (0x3000, 0.0, "PASS"),
            (0x3001, 0.0, "FAIL"),
            (0xB000, 0.0, "PASS"),
            (0xB001, 0.0, "FAIL"),
            (0x6401, 1024.4375, "FAIL"),
        ):
            result = compare(np.array([actual], dtype="<u2"), np.array([reference]))
            self.assertEqual(result["status"], expected)
            self.assertEqual(result["policy"], candidate.REFERENCE_POLICY)
            self.assertEqual(result["absolute_tolerance"], 0.125)
        with self.assertRaisesRegex(ValueError, "nonfinite comparison"):
            compare(np.array([0x7C00], dtype="<u2"), np.array([0.0]))
        with self.assertRaisesRegex(ValueError, "nonfinite comparison"):
            compare(np.array([0x0000], dtype="<u2"), np.array([math.nan]))
        for _, reference_hex in WITNESS_REFERENCES:
            reference = float.fromhex(reference_hex)
            result = compare(
                np.array([nearest_word(reference)], dtype="<u2"), np.array([reference])
            )
            self.assertEqual(result["status"], "FAIL")

    def test_result_and_specification_bind_scope_sources_and_inputs(self):
        spec = json.loads((candidate.ROOT / candidate.SOURCE_PATHS[0]).read_text())
        result = candidate.evaluate_layer_final_output(
            actual_fp16_bits=0x8000, reference_binary64_hex=(-0.0).hex()
        )
        for key in ("profile_id", "boundary", "precision", "reference_policy"):
            self.assertEqual(result[key], spec[key])
        self.assertEqual(result["profile_version"], spec["version"])
        self.assertEqual(result["actual_fp16_bits"], "8000")
        self.assertEqual(result["reference_binary64_hex"], "-0x0.0p+0")
        self.assertEqual(result["excess_budget"], "1/8")
        self.assertEqual([source["path"] for source in result["sources"]], list(candidate.SOURCE_PATHS))
        self.assertTrue(all(len(source["sha256"]) == 64 for source in result["sources"]))

    def test_cli_scalar_only_exit_codes(self):
        for bits, expected, reason in (
            ("3000", 0, None), ("3001", 1, None),
            ("7c00", 2, "nonfinite_actual_fp16"),
            ("10000", 2, "invalid_actual_fp16_encoding"),
        ):
            completed = subprocess.run(
                [sys.executable, "-B", "-m", "ace3.model.candidates.binary64_fp16_excess_v1",
                 "--actual-fp16-bits", bits, "--reference-binary64-hex", "0x0.0p+0"],
                cwd=candidate.ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, expected, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["profile_id"], candidate.PROFILE_ID)
            if reason:
                self.assertEqual(result["reason"], reason)
                self.assertEqual(result["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
