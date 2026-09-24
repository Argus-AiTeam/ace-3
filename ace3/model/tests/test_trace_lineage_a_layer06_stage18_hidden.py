import unittest
from fractions import Fraction

from trace_lineage_a_layer06_stage18_hidden import (
    ORDERS, deltas, denominator_split, hidden_factorial, project_cases, reference_add,
)


class Layer06HiddenTests(unittest.TestCase):
    def test_residual_sign_ties_subnormal_and_cancellation(self):
        self.assertEqual(reference_add(
            [0, 0x8000, 1, 0x3c00, 0x3c01, 0xbc00],
            [0x8000, 0x8000, 1, 0x1000, 0x1000, 0x3c00]),
            [0, 0x8000, 2, 0x3c00, 0x3c02, 0])

    def test_invalid_residual_inputs_are_not_silently_admitted(self):
        for a, b in (([], []), ([0], []), ([0x7c00], [0]),
                     ([0x7e00], [0]), ([-1], [0]), ([65536], [0]),
                     ([0x7bff], [0x7bff])):
            with self.assertRaises(ValueError):
                reference_add(a, b)

    def test_full_denominator_moves_unchanged_numerator(self):
        actual = {12: [0x4000, 0x3c00], 17: [0, 0]}
        reference = {12: [0x3c00, 0x3c00], 17: [0, 0]}
        for stages in (actual, reference):
            stages[18] = reference_add(stages[12], stages[17])
        hidden, norms, reductions, rows = hidden_factorial(actual, reference, [0x3c00] * 2)
        self.assertEqual(hidden["A"][1], hidden["R"][1])
        self.assertNotEqual(norms["A"][1], norms["R"][1])
        parts = denominator_split(hidden, norms, reductions, [0x3c00] * 2, 1)
        term = parts["stage12_first"]["stage12_at_reference_stage17"]
        self.assertEqual(Fraction(term["numerator_change"]), 0)
        self.assertNotEqual(Fraction(term["denominator_change"]), 0)
        self.assertEqual(sum(Fraction(term[k]) for k in (
            "numerator_change", "denominator_change",
            "rounding_and_binary64_operation_remainder")), Fraction(term["rounded_total"]))
        self.assertEqual(rows[1]["hidden_components_q24"]["stage12"], 0)

    def test_two_orders_preserve_cancelling_interaction(self):
        actual = {12: [0x4000, 0x3c00], 17: [0x3c00, 0x4000]}
        reference = {12: [0x3c00, 0x3c00], 17: [0, 0]}
        for stages in (actual, reference):
            stages[18] = reference_add(stages[12], stages[17])
        _, norms, _, rows = hidden_factorial(actual, reference, [0x3c00] * 2)
        self.assertTrue(any(row["norm_interaction_q24"] != 0 for row in rows))
        for i in range(2):
            parts = [deltas(norms, order, i) for order in ORDERS]
            self.assertEqual(sum(parts[0].values()), sum(parts[1].values()))

    def test_native_gemm_weighted_factorial_all_nibble_lanes(self):
        norms = {name: [value] * 128 for name, value in
                 (("A", 0x3c04), ("AA", 0x3c04), ("AR", 0x3c02),
                  ("RA", 0x3c03), ("RR", 0x3c00), ("R", 0x3c00))}
        tensors = {"qweight": [0x76543210] * 128, "qzeros": [0x11111111],
                   "scales": [0x3800] * 8, "bias": [0] * 8}
        for channel in range(8):
            raw, coefficients, totals = project_cases(norms, tensors, channel)
            expected = (((channel % 2) * 4 + channel // 2) - 1) * (1 << 23)
            self.assertEqual(coefficients, [expected] * 128)
            self.assertEqual(set(raw), set(norms))
            for transitions in ORDERS.values():
                self.assertEqual(sum(totals[a] - totals[b] for _, a, b in transitions),
                                 totals["A"] - totals["R"])
