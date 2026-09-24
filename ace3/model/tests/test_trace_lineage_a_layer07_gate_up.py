import unittest

from trace_lineage_a_layer07_gate_up import native_projection_pair, oracle_pair


class GateUpReconstructionTests(unittest.TestCase):
    def test_native_lanes_and_no_zero_plus_one(self):
        actual = [0x3c00] + [0] * 127
        reference = [0x3800] + [0] * 127
        tensors = {"qweight": [0x76543210] * 128,
                   "qzeros": [0x11111111], "scales": [0x3c00] * 8}
        for lane, weight in enumerate((0, 4, 1, 5, 2, 6, 3, 7)):
            pair = native_projection_pair(actual, reference, tensors, lane)
            self.assertEqual(pair["actual_q48"], (weight - 1) << 48)
            self.assertEqual(pair["reference_q48"], (weight - 1) << 47)
            for label, oracle in zip(("actual", "reference"),
                                     oracle_pair(actual, reference, tensors, lane)):
                self.assertEqual(pair[f"{label}_q48"], oracle[0])
                self.assertEqual(pair[f"{label}_bits"], oracle[1])

    def test_exact_cross_group_cancellation(self):
        actual = [0] * 256
        actual[0], actual[1], actual[128] = 0x3c00, 0x0001, 0xbc00
        tensors = {"qweight": [0x11111111] * 256,
                   "qzeros": [0, 0], "scales": [0x3c00] * 16}
        pair = native_projection_pair(actual, [0] * 256, tensors, 0)
        self.assertEqual(pair["actual_groups_q48"], [(1 << 48) + (1 << 24), -(1 << 48)])
        self.assertEqual(pair["actual_bits"], 0x0001)
        self.assertEqual(pair["reference_bits"], 0)

    def test_geometry_and_nonfinite_fail_explicitly(self):
        tensors = {"qweight": [0] * 128, "qzeros": [0], "scales": [0x3c00] * 8}
        with self.assertRaises(ValueError):
            native_projection_pair([0] * 127, [0] * 127, tensors, 0)
        with self.assertRaises(ValueError):
            native_projection_pair([0x7c00] + [0] * 127, [0] * 128, tensors, 0)
        with self.assertRaises(ValueError):
            native_projection_pair([0] * 128, [0] * 128, tensors, 8)


if __name__ == "__main__":
    unittest.main()
