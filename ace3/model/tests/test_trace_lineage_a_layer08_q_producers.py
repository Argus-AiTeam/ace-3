import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_lineage_a_layer08_q_producers import exact_projection, rounded_dot, units


class ExactQProjectionTest(unittest.TestCase):
    def test_native_lanes_zero_policy_and_single_round_bias(self):
        tensors = {
            "qweight": np.full((128, 1), 0x76543210, dtype="<u4"),
            "qzeros": np.full((1, 1), 0x22222222, dtype="<u4"),
            "scales": np.full((1, 8), 0x2000, dtype="<u2"),
            "bias": np.full(8, 0x3c00, dtype="<u2"),
        }
        for channel, physical in enumerate((0, 4, 1, 5, 2, 6, 3, 7)):
            bit, total, weight = exact_projection([0x3c00] * 128, tensors, channel)
            expected = 1 + 128 * (physical - 2) / 128
            self.assertEqual(units(bit) / 2**24, expected)
            self.assertEqual(total, int(expected * 2**48))
            self.assertEqual(weight, [(physical - 2) * units(0x2000)] * 128)

    def test_q48_ties_and_negative_subnormal(self):
        for lower, upper, expected in ((0, 1, 0), (1, 2, 2),
                                       (0x3bff, 0x3c00, 0x3c00),
                                       (0x3c00, 0x3c01, 0x3c00)):
            total = (units(lower) + units(upper)) << 23
            self.assertEqual(rounded_dot(total << 3), expected)
            self.assertEqual(rounded_dot(-(total << 3)), expected | 0x8000)


if __name__ == "__main__":
    unittest.main()
