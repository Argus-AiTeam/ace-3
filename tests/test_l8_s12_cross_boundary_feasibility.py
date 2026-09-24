"""Scalar arithmetic controls for the retained-only feasibility diagnostic."""

from fractions import Fraction
import unittest

from ace3.model.candidates.diagnose_l8_s12_cross_boundary import gate, half, independent_add
from ace3.model.fp16_adaptation_oracle import residual_add


class CrossBoundaryTests(unittest.TestCase):
    def test_exact_decoding_and_invalid_rejection(self):
        self.assertEqual(half(1), Fraction(1, 1 << 24))
        self.assertEqual(half(0x8001), -Fraction(1, 1 << 24))
        self.assertEqual(half(0x7bff), 65504)
        for bits in (-1, 65536, 0x7c00, 0xfc00, 0x7e00):
            with self.assertRaises(ValueError):
                half(bits)

    def test_unchanged_gate_boundaries(self):
        self.assertTrue(gate(0x3000, 0)["passed"])  # Inclusive absolute 1/8.
        self.assertFalse(gate(0x3001, 0)["passed"])
        self.assertTrue(gate(0x662c, 0x662d)["passed"])
        self.assertFalse(gate(0x662b, 0x662d)["passed"])
        self.assertEqual(gate(0x8000, 0)["ulp"], 0)
        self.assertEqual(gate(1, 0)["relative_error"], "1/1024")

    def test_native_residual_matches_independent_rounding(self):
        pairs = [(0, 0x8000), (0x8000, 0x8000), (1, 1), (1, 0x8001),
                 (0x03ff, 1), (0x3c00, 0x1000), (0x3c01, 0x1000),
                 (0x7bff, 0x7bff), (0xfbff, 0xfbff), (0x7bff, 0x4c00),
                 (0x662b, 0xb1f0), (0x662d, 0xb1f0)]
        for a, b in pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(residual_add(a, b), independent_add(a, b))


if __name__ == "__main__":
    unittest.main()
