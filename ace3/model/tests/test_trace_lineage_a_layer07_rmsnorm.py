import unittest

from fp16_adaptation_oracle import rmsnorm
from trace_lineage_a_layer07_rmsnorm import (
    floor_sqrt, mathematical_outputs, normalize, reconstruct,
)


class RMSNormReconstructionTests(unittest.TestCase):
    def test_integer_square_root_boundaries(self):
        for root in (0, 1, 2, 17, 2**24 - 1, 2**40):
            for value in (max(0, root * root - 1), root * root, root * root + 1):
                got = floor_sqrt(value)
                self.assertLessEqual(got * got, value)
                self.assertGreater((got + 1) ** 2, value)

    def test_oracle_general_finite_vectors(self):
        values = [0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3555, 0x3c00,
                  0xbc00, 0x4000, 0xc200, 0x7bff, 0xfbff]
        for count in (1, 2, 7, 128, 896):
            activations = [values[(i * 5 + count) % len(values)] for i in range(count)]
            weights = [[0x3c00, 0xbc00, 0x3800, 0, 0x8000][i % 5] for i in range(count)]
            got, reduction = reconstruct(activations, weights)
            expected, mean, root = rmsnorm(activations, weights)
            self.assertTrue(all(not invalid and not sat for _, invalid, sat in expected))
            self.assertEqual(got, [x[0] for x in expected])
            self.assertEqual((reduction["mean_q48"], reduction["rms_q24"]), (mean, root))

    def test_signed_zero_and_subnormal(self):
        self.assertEqual(normalize([0, 0x8000, 0], [0xbc00, 0x3c00, 0x3c00], 1),
                         [0x8000, 0x8000, 0])
        self.assertEqual(normalize([1, 0x8001], [0x3c00, 0x3c00], 1 << 24),
                         [1, 0x8001])

    def test_quotient_ties_even(self):
        self.assertEqual(normalize([1, 3, 5, 7], [0x3c00] * 4, 2 << 24), [0, 2, 2, 4])

    def test_fp16_midpoint_ties_even(self):
        self.assertEqual(normalize([0x3c01], [0x3a00], 1 << 24), [0x3a02])
        self.assertEqual(normalize([0x3c03], [0x3a00], 1 << 24), [0x3a04])

    def test_mathematical_signs_and_zeros(self):
        self.assertEqual(mathematical_outputs([0x3c00, 0xbc00], [0x3c00, 0x3c00]),
                         [0x3c00, 0xbc00])
        self.assertEqual(mathematical_outputs([0, 0x8000], [0xbc00, 0xbc00]),
                         [0x8000, 0])

    def test_nonfinite_and_bad_geometry_rejected(self):
        for value in (0x7c00, 0xfc00, 0x7e00, -1, 0x10000):
            with self.assertRaises(ValueError):
                reconstruct([value], [0x3c00])
            with self.assertRaises(ValueError):
                reconstruct([0x3c00], [value])
        with self.assertRaises(ValueError):
            reconstruct([], [])
        with self.assertRaises(ValueError):
            reconstruct([0], [])
        with self.assertRaises(ValueError):
            normalize([0], [0], 0)


if __name__ == "__main__":
    unittest.main()
