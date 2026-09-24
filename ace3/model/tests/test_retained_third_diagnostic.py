import unittest

from layer08_diagnostic_retained_third import dot_accounting, half, project


class RetainedThirdDiagnosticTests(unittest.TestCase):
    def test_signed_dot_decomposition_with_interaction(self):
        row = dot_accounting([half(2)] * 64, [half(-3)] * 64,
                             [half(1)] * 64, [half(-2)] * 64)
        self.assertEqual(row["totals"]["actual_bits"], f"{half(-48):04x}")
        self.assertEqual(row["totals"]["reference_bits"], f"{half(-16):04x}")
        self.assertEqual(row["score_units"]["q_delta_q48"], "-16")
        self.assertEqual(row["score_units"]["k_delta_q48"], "-8")
        self.assertEqual(row["score_units"]["interaction_q48"], "-8")

    def test_nonfinite_and_wrong_geometry_rejected(self):
        for q in ([0] * 63, [0x7c00] * 64):
            with self.assertRaises(ValueError):
                dot_accounting(q, [0] * 64, [0] * 64, [0] * 64)

    def test_native_awq_order_and_no_zero_plus_one(self):
        tensors = dict(qweight=[0x76543210] * 128, qzeros=[0x11111111],
                       scales=[half(1)] * 8, bias=[half(0)] * 8)
        for channel, nibble in enumerate((0, 4, 1, 5, 2, 6, 3, 7)):
            actual, _, terms = project([half(2)] + [0] * 127, tensors, channel)
            self.assertEqual(actual, half(2 * (nibble-1)))
            self.assertEqual(terms[0]["qzero"], 1)
