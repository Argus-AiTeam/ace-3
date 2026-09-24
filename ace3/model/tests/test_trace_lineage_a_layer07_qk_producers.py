import unittest

from trace_lineage_a_layer07_qk_producers import (
    DECODER_SINGLE_ROUND_PRODUCERS, projection, rotations, score_terms, split,
)


class QKProducerTests(unittest.TestCase):
    def test_telescoping_signed_terms(self):
        self.assertEqual(split([0x3c00, 0, 0xbc00], ("first", "second")),
                         {"first": 1 << 24, "second": 1 << 24})

    def test_interaction_is_not_q_or_k_drift(self):
        rows = score_terms([0x4000]*64, [0x3c00]*64, [0x4200]*64, [0x3c00]*64)
        self.assertEqual(rows[0]["q_dot_q48"], 1 << 48)
        self.assertEqual(rows[0]["k_dot_q48"], 2 << 48)
        self.assertEqual(rows[0]["interaction_dot_q48"], 2 << 48)

    def test_all_native_lanes_zero_delta(self):
        tensors = {"qweight": [0x76543210]*128, "qzeros": [0x76543210],
                   "scales": [0x3c00]*8, "bias": [0xbc00]*8}
        for lane in range(8):
            for single in (False, True):
                self.assertEqual(projection([0x3c00]*128, tensors, lane, single),
                                 (0xbc00, 0xbc00))

    def test_bias_rounding_boundaries_are_distinct(self):
        tensors = {"qweight": [0]*128, "qzeros": [0],
                   "scales": [0x3c00]*8, "bias": [0x1000]*8}
        tensors["qweight"][:2] = [1, 1]
        activation = [0x3c00, 0x1000] + [0]*126
        self.assertEqual(projection(activation, tensors, 0, False), (0x3c00, 0x3c01))
        self.assertEqual(projection(activation, tensors, 0, True), (0x3c01, 0x3c01))
        for producer, expected in (("q", 0x3c01), ("k", 0x3c01), ("v", 0x3c00)):
            actual, _ = projection(activation, tensors, 0,
                                   producer in DECODER_SINGLE_ROUND_PRODUCERS)
            self.assertEqual(actual, expected)

    def test_identity_rope_and_signed_inputs(self):
        for pair in ([0x3c00, 0xc000], [0x0001, 0x8001]):
            self.assertEqual(rotations(pair, (0x3c00, 0), 0, 0), (pair, pair, pair))

    def test_invalid_operands_and_geometry(self):
        with self.assertRaises(ValueError):
            score_terms([0]*63, [0]*64, [0]*64, [0]*64)
        with self.assertRaises(ValueError):
            split([0, 0x7c00], ("nonfinite",))
        with self.assertRaises(ValueError):
            rotations([0x7c00, 0], (0x3c00, 0), 0, 0)
