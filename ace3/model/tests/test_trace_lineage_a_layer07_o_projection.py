import unittest

from trace_lineage_a_layer07_o_projection import (
    native_projection_pair, oracle_pair, projection_components, residual_components,
)


class OProjectionBoundaryTests(unittest.TestCase):
    def test_all_native_lanes_and_no_zero_adjustment(self):
        shifts = (0, 16, 4, 20, 8, 24, 12, 28)
        packed = sum(lane << shift for lane, shift in enumerate(shifts))
        tensors = {"qweight": [packed] * 128, "qzeros": [0x33333333],
                   "scales": [0x2000] * 8}
        for lane in range(8):
            result = native_projection_pair([0x3c00] * 128, [0x3800] * 128, tensors, lane)
            self.assertEqual(result["actual_q48"], (lane - 3) * (1 << 48))
            self.assertEqual(result["reference_q48"], (lane - 3) * (1 << 47))
            for label, check in zip(("actual", "reference"),
                                    oracle_pair([0x3c00] * 128, [0x3800] * 128, tensors, lane)):
                self.assertEqual(check[:4], (result[f"{label}_q48"],
                                            result[f"{label}_bits"], False, False))

    def test_groups_preserve_unrounded_cancellation(self):
        tensors = {"qweight": [0x44444444] * 256, "qzeros": [0x33333333, 0x55555555],
                   "scales": [0x0001] * 16}
        result = native_projection_pair([0x0001] * 256, [0x8001] * 256, tensors, 5)
        self.assertEqual(result["actual_groups_q48"], [128, -128])
        self.assertEqual(result["reference_groups_q48"], [-128, 128])
        self.assertEqual(result["actual_bits"], 0)
        self.assertEqual(result["reference_bits"], 0)

    def test_decomposition_keeps_reference_and_local_separate(self):
        measured = {"actual_q48": 1 << 48, "reference_q48": 1 << 47,
                    "actual_bits": 0x3c00, "reference_bits": 0x3800}
        exact = projection_components(measured, 0x3c00, 0x3800)
        self.assertEqual(exact["attention_input_delta_q48"], 1 << 47)
        self.assertEqual(exact["local_projection_delta_q48"], 0)
        local = projection_components(measured, 0x3c01, 0x3800)
        self.assertEqual(local["local_projection_delta_q48"], 1 << 38)
        reference = projection_components(measured, 0x3c00, 0x3801)
        self.assertEqual(reference["reference_recurrence_delta_q48"], -(1 << 37))

    def test_projection_rounding_is_explicit(self):
        measured = {"actual_q48": (1 << 48) + (1 << 36),
                    "reference_q48": 1 << 48, "actual_bits": 0x3c00, "reference_bits": 0x3c00}
        parts = projection_components(measured, 0x3c00, 0x3c00)
        self.assertEqual(parts["projection_rounding_delta_q48"], -(1 << 36))
        self.assertEqual(sum(parts.values()), 0)

    def test_retained_residual_witness(self):
        parts = residual_components(0xb37e, 0xb382, 0x3190, 0x318d, 0xabb8, 0xabd4)
        self.assertEqual(parts, {"incoming_hidden_delta_q24": 8192,
                                 "projection_delta_q24": 6144,
                                 "residual_rounding_delta_q24": 0,
                                 "stage12_delta_q24": 14336})

    def test_reject_wrong_residual(self):
        with self.assertRaisesRegex(ValueError, "residual reconstruction"):
            residual_components(0xb37e, 0xb382, 0x3190, 0x318d, 0xabb9, 0xabd4)

    def test_nonfinite_and_geometry_rejected(self):
        tensors = {"qweight": [0] * 128, "qzeros": [0], "scales": [0x3c00] * 8}
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            native_projection_pair([0x7c00] * 128, [0] * 128, tensors, 0)
        with self.assertRaisesRegex(ValueError, "geometry"):
            native_projection_pair([0] * 127, [0] * 127, tensors, 0)


if __name__ == "__main__":
    unittest.main()
