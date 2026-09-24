import sys
import unittest
from decimal import Decimal, localcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fp16_adaptation_oracle import silu_gate_exp
from trace_lineage_a_layer07_silu import (
    accurate_silu, classification, mathematical_silu, nearest_decimal, rne_ratio,
)


class ClassificationTest(unittest.TestCase):
    def test_reference_only_difference_is_not_implementation_defect(self):
        result = classification({"actual": [], "reference": [782, 932]})
        self.assertEqual(result["status"], "RETAINED_SILU_RECONSTRUCTED_NOT_REPAIRED")
        self.assertFalse(result["causal_local_implementation_defect_proven"])

    def test_actual_difference_remains_a_discrepancy(self):
        for reference in ([], [782]):
            result = classification({"actual": [3631], "reference": reference})
            self.assertEqual(result["status"], "LOCAL_SILU_DISCREPANCY_NOT_REPAIRED")
            self.assertTrue(result["causal_local_implementation_defect_proven"])


class SiLUDiagnosticTest(unittest.TestCase):
    def test_signed_ties(self):
        for numerator, expected in ((1, 0), (3, 2), (5, 2), (7, 4), (9, 4)):
            self.assertEqual(rne_ratio(numerator, 2), expected)
            self.assertEqual(rne_ratio(-numerator, 2), -expected)
        with self.assertRaises(ValueError):
            rne_ratio(1, 0)

    def test_fp16_rounding_boundaries(self):
        with localcontext() as ctx:
            ctx.prec = 80
            self.assertEqual(nearest_decimal(Decimal(1) / (1 << 25)), 0)
            self.assertEqual(nearest_decimal(Decimal(3) / (1 << 25)), 2)
            self.assertEqual(nearest_decimal(Decimal(1) + Decimal(1) / 2048), 0x3c00)
            self.assertEqual(nearest_decimal(-Decimal(1) - Decimal(3) / 2048), 0xbc02)
            self.assertEqual(nearest_decimal(Decimal(65504)), 0x7bff)
            self.assertEqual(nearest_decimal(Decimal(0), 0x8000), 0x8000)
            for value in (Decimal("NaN"), Decimal("Infinity"), Decimal(65505)):
                with self.assertRaises(ValueError):
                    nearest_decimal(value)

    def test_source_operator_against_existing_oracle(self):
        gates = (0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3555,
                 0xb555, 0x398b, 0xb98c, 0x3c00, 0xbc00, 0x4c00,
                 0xcc00, 0x53e0, 0xd3e0)
        for gate in gates:
            for up in (0, 0x8000, 1, 0x8001, 0x3800, 0xb800, 0x3c00, 0xbc00):
                with self.subTest(gate=gate, up=up):
                    self.assertEqual(
                        (accurate_silu(gate, up)["bits"], False, False),
                        silu_gate_exp(gate, up),
                    )
        for invalid in (0x7c00, 0xfc00, 0x7e00, -1, 0x10000):
            with self.assertRaises(ValueError):
                accurate_silu(invalid, 0x3c00)
            with self.assertRaises(ValueError):
                accurate_silu(0x3c00, invalid)

    def test_independent_mathematical_identity(self):
        with localcontext() as ctx:
            ctx.prec = 80
            positive = mathematical_silu(0x3c00, 0x3c00)
            negative = mathematical_silu(0xbc00, 0x3c00)
            self.assertLess(abs(positive - negative - 1), Decimal("1e-75"))
            self.assertEqual(nearest_decimal(positive), 0x39d9)
            self.assertEqual(nearest_decimal(negative), 0xb44e)
            self.assertEqual(mathematical_silu(0, 0x3c00), 0)


if __name__ == "__main__":
    unittest.main()
