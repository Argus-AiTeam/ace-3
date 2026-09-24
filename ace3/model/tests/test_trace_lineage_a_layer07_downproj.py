import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnose_lineage_a_position3_boundary import units
from trace_lineage_a_layer07_downproj import projection_terms, residual, round_q48


class DownProjectionDiagnosticTest(unittest.TestCase):
    def test_rounding_and_residual(self):
        for bits in (0, 1, 0x8001, 0x03ff, 0x0400, 0x3bff, 0x3c00, 0x7bff):
            self.assertEqual(round_q48(units(bits) << 24), bits)
        for lower in (0, 1, 0x03ff, 0x3bff, 0x3c00):
            midpoint = (units(lower) + units(lower + 1)) << 23
            even = lower if not lower & 1 else lower + 1
            self.assertEqual(round_q48(midpoint), even)
            self.assertEqual(round_q48(-midpoint), even | (0x8000 if even else 0))
        self.assertEqual(residual(0x3c00, 0xbc00), 0)
        self.assertEqual(residual(0xabb8, 0x3405), 0x321c)
        self.assertEqual(residual(0xabd4, 0x340a), 0x321f)
        with self.assertRaises(ValueError):
            round_q48((units(0x7bff) << 24) + 1)
        with self.assertRaises(ValueError):
            residual(0x7c00, 0)

    def test_native_lanes_and_signed_terms(self):
        for channel, shift in enumerate((0, 16, 4, 20, 8, 24, 12, 28)):
            activations = [0x3c00] * 4864
            reference = [0] * 4864
            weights = [3 << shift] * (4864 * 112)
            zeros = [5 << shift] * (38 * 112)
            scales = [0x3800] * (38 * 896)
            terms = projection_terms(activations, reference, weights, zeros, scales, channel)
            self.assertTrue(all(t["signed_delta"] == -2 and t["actual_term_q48"] == -(1 << 48)
                                and t["reference_term_q48"] == 0 for t in terms))
            self.assertEqual(terms[128]["group"], 1)
            self.assertEqual(terms[-1]["group"], 37)


if __name__ == "__main__":
    unittest.main()
