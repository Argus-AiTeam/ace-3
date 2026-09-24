import unittest

from trace_lineage_a_layer07_attention_value import compose, decompose, nearest_f16


class AttentionValueBoundaryTests(unittest.TestCase):
    def test_cancellation_and_negative_values(self):
        self.assertEqual(compose([0x3800, 0x3800], [0x3c00, 0xbc00]), (0, 0))
        self.assertEqual(compose([0x3c00], [0xbc00]), (-(1 << 48), 0xbc00))

    def test_q24_ties_and_signed_zero(self):
        self.assertEqual(compose([0x3800], [0x0001]), (1 << 23, 0))
        self.assertEqual(compose([0x3800], [0x8001]), (-(1 << 23), 0))
        self.assertEqual(compose([0x3800], [0x0003]), (3 << 23, 2))

    def test_fp16_even_midpoints(self):
        self.assertEqual(nearest_f16((1 << 24) + (1 << 13), 1), 0x3c00)
        self.assertEqual(nearest_f16((1 << 24) + 3 * (1 << 13), 1), 0x3c02)
        self.assertEqual(nearest_f16(-((1 << 24) + (1 << 13)), 1), 0xbc00)

    def test_probability_v_and_interaction_are_separate(self):
        row = decompose([0x3800, 0x3800], [0x3400, 0x3400],
                        [0x4000, 0x4000], [0x3c00, 0x3c00], 0x4000, 0x3800)
        parts = row["components_q48"]
        self.assertEqual(parts["probability"], 1 << 47)
        for key in ("historical_v", "current_v", "historical_interaction", "current_interaction"):
            self.assertEqual(parts[key], 1 << 46)
        self.assertEqual(parts["local_av"], 0)
        self.assertEqual(parts["reference_recurrence"], 0)

    def test_local_and_reference_discrepancies_not_absorbed(self):
        row = decompose([0x3c00], [0x3c00], [0x3c00], [0x3c00], 0x3c01, 0x3c01)
        self.assertEqual(row["components_q48"]["local_av"], 1 << 38)
        self.assertEqual(row["components_q48"]["reference_recurrence"], -(1 << 38))

    def test_invalid_operands_and_geometry_rejected(self):
        for p, v in (([0xbc00], [0x3c00]), ([0x7c00], [0x3c00]),
                     ([0], [0x7e00]), ([], []), ([0], [])):
            with self.assertRaises(ValueError):
                compose(p, v)
