import unittest

from trace_lineage_a_layer07_stage00_inputs import (
    PARTS, differences, norm_measurement, reference_norm, split_bits, weighted_channel,
)
from trace_lineage_a_layer07_rmsnorm import reconstruct


class Stage00InputsTests(unittest.TestCase):
    def test_signed_norm_and_independent_recurrence(self):
        self.assertEqual(reference_norm([0x3c00, 0xbc00], [0x3c00, 0x3c00]),
                         [0x3c00, 0xbc00])
        self.assertEqual(reference_norm([0, 0x8000], [0xbc00, 0x3c00]),
                         [0x8000, 0x8000])

    def test_local_policy_hidden_split(self):
        hidden = [0x3c00, 0xbc00, 1, 0x8000] * 224
        ref_hidden = [0x3c01, 0xbc00, 1, 0x8000] * 224
        weights = [0x3c00] * 896
        retained = reconstruct(hidden, weights)[0]
        reference = reference_norm(ref_hidden, weights)
        chains, rows, summary = norm_measurement(
            hidden, ref_hidden, weights, retained, reference)
        self.assertEqual(summary["local_bit_differences"], [])
        self.assertEqual(summary["reference_recurrence_bit_differences"], [])
        self.assertEqual(len(summary["hidden_bit_differences"]), 224)
        for i, row in enumerate(rows):
            self.assertEqual(row["components_q24"], split_bits([x[i] for x in chains]))

    def test_exact_weighted_native_gemm_closure(self):
        chains = [[value] * 128 for value in (0x3c04, 0x3c03, 0x3c02, 0x3c01, 0x3c00)]
        tensors = {"qweight": [0x76543210] * 128, "qzeros": [0x11111111],
                   "scales": [0x3800] * 8, "bias": [0] * 8}
        for channel in range(8):
            raw, rows, parts = weighted_channel(chains, tensors, channel)
            self.assertEqual(len(raw), 5)
            self.assertEqual(len(rows), 128)
            self.assertEqual(set(parts), set(PARTS))
            self.assertEqual(sum(parts.values()), sum(r["delta_q48"] for r in rows))
            for row in rows:
                self.assertEqual(sum(row["components_q48"].values()), row["delta_q48"])

    def test_nonfinite_or_invalid_geometry_rejected(self):
        for bad in (0x7c00, 0x7e00, -1, 0x10000):
            with self.assertRaises(ValueError):
                reference_norm([bad], [0x3c00])
        with self.assertRaises(ValueError):
            reference_norm([], [])
        with self.assertRaises(ValueError):
            split_bits([0x3c00] * 4)
        with self.assertRaises(ValueError):
            differences([0], [])

    def test_signed_zero_is_not_a_numerical_delta(self):
        self.assertEqual(differences([0], [0x8000]), [0])
        self.assertEqual(sum(split_bits([0, 0x8000, 0, 0x8000, 0]).values()), 0)

    def test_cancelling_components_are_not_erased(self):
        parts = split_bits([0x3c00, 0x4000, 0xbc00, 0, 0x3c00])
        self.assertEqual(sum(parts.values()), 0)
        self.assertGreater(sum(abs(x) for x in parts.values()), 0)
