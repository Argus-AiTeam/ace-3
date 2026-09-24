import sys
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from diagnose_lineage_a_position3_boundary import (
    accepts, decode_trace, rounded_dot, rounding_margin, score_measurement, units,
)


class LineageABoundaryTest(unittest.TestCase):
    def test_finite_binary16_units(self):
        for bits, expected in ((0, 0), (0x8000, 0), (1, 1), (0x8001, -1),
                               (0x0400, 1024), (0x3c00, 1 << 24),
                               (0x7bff, 65504 << 24)):
            self.assertEqual(units(bits), expected)
        for bits in (-1, 0x10000, 0x7c00, 0xfc00, 0x7e00):
            with self.assertRaises(ValueError):
                units(bits)

    def test_unchanged_gate(self):
        self.assertTrue(accepts(0x3000, 0))  # Inclusive 0.125 absolute budget.
        self.assertFalse(accepts(0x3001, 0))
        self.assertTrue(accepts(0x62a7, 0x62a8))
        self.assertFalse(accepts(0x62a6, 0x62a8))
        self.assertTrue(accepts(0x8000, 0))
        # 0.5/500 is exactly 0.001: the relative comparison is strict.
        self.assertFalse(accepts(0x5fd2, 0x5fd0))

    def test_exact_dot_rounding_ties(self):
        for lower, upper, expected in ((0, 1, 0), (1, 2, 2),
                                       (0x3bff, 0x3c00, 0x3c00),
                                       (0x3c00, 0x3c01, 0x3c00)):
            midpoint = (units(lower) + units(upper)) << 26
            self.assertEqual(rounded_dot(midpoint), expected)
            if expected:
                self.assertEqual(rounded_dot(-midpoint), expected | 0x8000)
        with self.assertRaises(ValueError):
            rounded_dot((units(0x7bff) << 27) + 1)

    def test_margin_and_attention_order(self):
        margin = rounding_margin(units(0x62a6) << 27, 0x62a6, 0x62a8)
        self.assertEqual(margin["nearest_passing_bits"], "62a7")
        self.assertFalse(margin["boundary_inclusive"])
        self.assertEqual(margin["required_signed_dot_change"]["rational"], "1/4")
        data = "".join(f"00000308{p:04x}3c00\n" for _ in range(14) for p in range(4))
        self.assertEqual(decode_trace(data.encode("ascii"), 3)[8], [0x3c00] * 56)
        with self.assertRaises(ValueError):
            decode_trace(data.replace("000003080001", "000003080000").encode("ascii"), 3)

    def test_conditional_score_keeps_original_gate(self):
        failed = score_measurement(units(0x62a6) << 27, 0x62a8)
        self.assertEqual(failed["value"], 851)
        self.assertFalse(failed["within_unchanged_gate"])
        passed = score_measurement(units(0x62a7) << 27, 0x62a8)
        self.assertTrue(passed["within_unchanged_gate"])
        self.assertEqual(passed["exact_score"]["rational"], "1703/2")


if __name__ == "__main__":
    unittest.main()
